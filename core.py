import os
import io
import json
import re
import base64
from typing import List, Dict, Any, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from rank_bm25 import BM25Okapi

import pdfplumber
from pypdf import PdfReader

import httpx
from pydantic import BaseModel
import functools


# =========================
# Config via variáveis de ambiente
# =========================
DEFAULT_PROVIDER = os.getenv("LLM_PROVIDER", "sabia").lower()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
SABIA_API_KEY = os.getenv("SABIA_API_KEY", "")
SABIA_BASE_URL = os.getenv("SABIA_BASE_URL", "")

MODEL_GPT = os.getenv("MODEL_GPT", "gpt-4o-mini")
MODEL_SABIA = os.getenv("MODEL_SABIA", "sabia-3.1")

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1100))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
TOP_K = int(os.getenv("TOP_K", 6))
USE_BM25 = (str(os.getenv("USE_BM25", "true")).lower() == "true")


# =========================
# Modelos de dados (Pydantic)
# =========================
class CaseParty(BaseModel):
    role: str
    name: str
    person_type: str
    id: Optional[str] = None
    address: Optional[str] = None
    emails: Optional[List[str]] = None


class Claim(BaseModel):
    title: str
    type: str
    details: Optional[str] = None


class CaseValue(BaseModel):
    currency: str
    value: float


class CaseData(BaseModel):
    forum: str
    parties: List[CaseParty]
    facts_summary: str
    claims: List[Claim]
    case_value: Optional[CaseValue] = None
    urgency: bool = False


class AgentConfig(BaseModel):
    temperature: float = 0.4


class Agents(BaseModel):
    writer: AgentConfig
    reviewer_merit: AgentConfig
    reviewer_proc: AgentConfig
    reviewer_format: AgentConfig


class Attachment(BaseModel):
    filename: str
    content: str  # base64 do PDF


class StructuredInput(BaseModel):
    case_data: CaseData
    agents: Agents
    attachments: List[Attachment] = []


# =========================
# Utils
# =========================
def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").replace("\x00", "")).strip()


def chunk_text(text: str, size: int, overlap: int) -> List[str]:
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        end = min(len(words), start + size)
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = max(0, end - overlap)
    return chunks


def read_pdf_text(file_bytes: bytes) -> str:
    # 1) pypdf
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        txt = "\n".join((page.extract_text() or "") for page in reader.pages)
        txt = normalize_text(txt)
        if txt:
            return txt
    except Exception:
        pass

    # 2) pdfplumber
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            txt = "\n".join((pg.extract_text() or "") for pg in pdf.pages)
        txt = normalize_text(txt)
        if txt:
            return txt
    except Exception:
        pass

    return ""


# =========================
# VectorIndex (FAISS + BM25)
# =========================
class VectorIndex:
    def __init__(self, embedder_name: str):
        self.model = SentenceTransformer(embedder_name)
        self.index = None
        self.doc_meta: List[Dict[str, Any]] = []
        self.emb_dim = self.model.get_sentence_embedding_dimension()
        self.bm25 = None

    def _build_bm25(self):
        tokenized = [m["text"].lower().split() for m in self.doc_meta]
        self.bm25 = BM25Okapi(tokenized)

    def add(self, texts: List[str], metas: List[Dict[str, Any]]):
        embs = self.model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True
        ).astype(np.float32)
        if self.index is None:
            self.index = faiss.IndexFlatIP(self.emb_dim)
        self.index.add(embs)
        self.doc_meta.extend(metas)
        if USE_BM25:
            self._build_bm25()
        for i, m in enumerate(self.doc_meta):
            m["idx"] = i

    def search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        if not self.doc_meta or self.index is None:
            return []
        q = self.model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype(np.float32)
        D, I = self.index.search(q, min(top_k, len(self.doc_meta)))
        vec_res = []
        for score, idx in zip(D[0].tolist(), I[0].tolist()):
            if idx < 0:
                continue
            meta = dict(self.doc_meta[idx])
            meta["score_vec"] = float(score)
            vec_res.append(meta)

        if USE_BM25 and self.bm25 is not None:
            bm = self.bm25.get_scores(query.lower().split())
            bmin, bmax = float(np.min(bm)), float(np.max(bm))
            denom = (bmax - bmin) or 1.0
            bm_norm = (bm - bmin) / denom
            fused: Dict[int, Dict[str, Any]] = {}
            for r in vec_res:
                idx = r["idx"]
                r["score_fused"] = 0.5 * r["score_vec"] + 0.5 * float(bm_norm[idx])
                fused[idx] = r
            for idx in np.argsort(-bm_norm)[:top_k].tolist():
                if idx not in fused and idx < len(self.doc_meta):
                    r = dict(self.doc_meta[idx])
                    r["score_vec"] = 0.0
                    r["score_fused"] = float(bm_norm[idx])
                    fused[idx] = r
            return sorted(
                fused.values(), key=lambda x: x["score_fused"], reverse=True
            )[:top_k]
        return vec_res[:top_k]


# =========================
# LLM Providers
# =========================
class LLMMessage(BaseModel):
    role: str
    content: str


def sabia_chat(messages: List[LLMMessage], model: str, temperature: float, max_tokens: int) -> str:
    base = (SABIA_BASE_URL or "").strip()
    key = (SABIA_API_KEY or "").strip()
    if not base.startswith("http"):
        base = "https://" + base
    if not base or not key:
        raise RuntimeError("Configure SABIA_BASE_URL e SABIA_API_KEY.")
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [m.model_dump() for m in messages],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    with httpx.Client(timeout=120) as c:
        r = c.post(base, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        return json.dumps(data, ensure_ascii=False)


def openai_chat(messages: List[LLMMessage], model: str, temperature: float, max_tokens: int) -> str:
    key = (OPENAI_API_KEY or "").strip()
    if not key:
        raise RuntimeError("Configure OPENAI_API_KEY.")
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [m.model_dump() for m in messages],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    url = "https://api.openai.com/v1/chat/completions"
    with httpx.Client(timeout=120) as c:
        r = c.post(url, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
    return data["choices"][0]["message"]["content"]


def llm_chat(provider: str, messages: List[LLMMessage], temperature: float, max_tokens: int) -> str:
    if provider == "gpt":
        return openai_chat(messages, MODEL_GPT, temperature, max_tokens)
    return sabia_chat(messages, MODEL_SABIA, temperature, max_tokens)


# =========================
# Prompts (versões resumidas)
# =========================
PROMPT_WRITER = """
Você é um advogado sênior brasileiro. Redija uma PETIÇÃO extremamente detalhada, 
aprofundada e robusta, contendo:

1) Endereçamento completo
2) Qualificação detalhada das partes
3) Exposição minuciosa dos fatos, com contextualização jurídica
4) Fundamentação jurídica extensa, incluindo:
   - doutrina relevante,
   - jurisprudência atualizada dos Tribunais Superiores,
   - interpretação sistemática das normas.
5) Pedidos numerados com argumentação extensa
6) Valor da causa e justificativa
7) Requerimentos finais completos
8) Rol de documentos anexos

Use linguagem jurídica refinada e altamente técnica.
Desenvolva parágrafos longos e bem articulados.
Aumente a profundidade da análise jurídica.
Inclua fundamentos constitucionais, civis e processuais sempre que pertinente.
"""

PROMPT_REVIEW_MERIT = """
Você é um revisor jurídico focado em MÉRITO. Avalie criticamente a minuta.
Responda em JSON:
{
 "issues": ["..."],
 "suggested_fixes": ["..."],
 "quality_notes": {"clareza":0-10,"aderencia":0-10,"riscos":"..."}
}
Seja específico e traga fundamentos, súmulas e precedentes quando possível.
"""

PROMPT_REVIEW_PROC = """
Você é um revisor jurídico focado em PROCEDIMENTO/FORMALIDADES.
Verifique foro, qualificação, estrutura, pedidos, valor da causa e anexos.
Responda em JSON:
{
 "issues": ["..."],
 "suggested_fixes": ["..."],
 "quality_notes": {"conformidade":0-10,"riscos_processuais":"..."}
}
"""

PROMPT_REVIEW_FORMAT = """
Você é um revisor de FORMATAÇÃO. Ajuste a minuta para:
- Cabeçalhos claros
- Parágrafos bem separados
- Numeração adequada
- Preservar citações [Fonte: <doc> <chunk_id>]
Responda apenas com o TEXTO FINAL formatado, sem comentários.
"""


# =========================
# Função principal
# =========================
def generate_petition(
    structured: StructuredInput,
    top_k: int = TOP_K,
    provider: str = DEFAULT_PROVIDER,
) -> Dict[str, Any]:
    case = structured.case_data
    agents = structured.agents
    attachments = structured.attachments

    t_writer = agents.writer.temperature
    t_merit = agents.reviewer_merit.temperature
    t_proc = agents.reviewer_proc.temperature
    t_format = agents.reviewer_format.temperature

    vindex = VectorIndex(EMBEDDING_MODEL)
    anexos_nomes: List[str] = []
    warnings: List[str] = []

    # 1) Ingestão RAG
    if attachments:
        for att in attachments:
            try:
                b = base64.b64decode(att.content)
            except Exception:
                warnings.append(f"Falha ao decodificar base64 de: {att.filename}")
                continue

            txt = read_pdf_text(b)
            if not txt:
                warnings.append(f"Sem texto extraído de: {att.filename} (OCR não habilitado).")
                continue

            chunks = chunk_text(txt, CHUNK_SIZE, CHUNK_OVERLAP)
            metas = []
            for i, ch in enumerate(chunks):
                metas.append({
                    "doc_path": att.filename,
                    "chunk_id": f"{att.filename}#chunk={i}",
                    "text": ch,
                    "idx": len(vindex.doc_meta) + i
                })
            vindex.add([m["text"] for m in metas], metas)
            anexos_nomes.append(att.filename)

    # 2) Query para RAG
    claims_texts = []
    for c in case.claims:
        base = c.title
        if c.details:
            base += f": {c.details}"
        claims_texts.append(normalize_text(base))

    query = case.facts_summary + " " + " ".join(claims_texts)
    passages = vindex.search(query, top_k=top_k) if vindex.doc_meta else []

    # 3) Resumo textual
    plaintiffs = [p for p in case.parties if p.role.lower() == "plaintiff"]
    defendants = [p for p in case.parties if p.role.lower() == "defendant"]

    def fmt_party(p: CaseParty) -> str:
        tipo = "Pessoa Física" if p.person_type == "PF" else "Pessoa Jurídica"
        doc_id = p.id or "-"
        addr = p.address or "-"
        return f"{p.name} ({tipo})  CPF/CNPJ: {doc_id}  ENDEREÇO: {addr}"

    autores_txt = "\n".join(f"- {fmt_party(p)}" for p in plaintiffs) or "-"
    reus_txt = "\n".join(f"- {fmt_party(p)}" for p in defendants) or "-"

    if case.case_value:
        valor_causa_str = f"{case.case_value.currency} {case.case_value.value}"
    else:
        valor_causa_str = "-"

    resumo = f"""
FORO/COMARCA: {case.forum}
ÁREA: Não especificada

AUTORES:
{autores_txt}

RÉUS:
{reus_txt}

CAUSA DE PEDIR:
{case.facts_summary}

PEDIDOS:
- """ + "\n- ".join(claims_texts) + f"""

VALOR DA CAUSA: {valor_causa_str}

URGÊNCIA: {'SIM' if case.urgency else 'NÃO'}
"""

    passages_txt = "\n".join(
        [f"- [{p['chunk_id']}] '{p['doc_path']}': {p['text'][:800]}..." for p in passages]
    ) or "(Nenhum trecho recuperado)"

    # 4) Writer
    draft = llm_chat(
        provider,
        [
            LLMMessage(role="system", content=PROMPT_WRITER),
            LLMMessage(
                role="user",
                content=(
                    f"RESUMO E DADOS DO CASO:\n{resumo}\n\n"
                    f"TRECHOS RECUPERADOS (use se pertinente):\n{passages_txt}"
                )
            )
        ],
        temperature=t_writer,
        max_tokens=10000,
    )

    # 5) Reviewer Mérito
    rev_merit_raw = llm_chat(
        provider,
        [
            LLMMessage(role="system", content=PROMPT_REVIEW_MERIT),
            LLMMessage(role="user", content=draft),
        ],
        temperature=t_merit,
        max_tokens=1200,
    )
    try:
        rev_merit = json.loads(rev_merit_raw)
    except Exception:
        rev_merit = {
            "issues": [],
            "suggested_fixes": [],
            "quality_notes": {"raw": rev_merit_raw[:800]},
        }

    # 6) Reviewer Procedimento
    rev_proc_raw = llm_chat(
        provider,
        [
            LLMMessage(role="system", content=PROMPT_REVIEW_PROC),
            LLMMessage(role="user", content=draft),
        ],
        temperature=t_proc,
        max_tokens=1200,
    )
    try:
        rev_proc = json.loads(rev_proc_raw)
    except Exception:
        rev_proc = {
            "issues": [],
            "suggested_fixes": [],
            "quality_notes": {"raw": rev_proc_raw[:800]},
        }

    # 7) Reviewer Formatação
    payload_format = f"""
MINUTA:
{draft}

REVISOR MÉRITO:
{json.dumps(rev_merit, ensure_ascii=False)}

REVISOR PROCEDIMENTO:
{json.dumps(rev_proc, ensure_ascii=False)}
"""
    final_formatted = llm_chat(
        provider,
        [
            LLMMessage(role="system", content=PROMPT_REVIEW_FORMAT),
            LLMMessage(role="user", content=payload_format),
        ],
        temperature=t_format,
        max_tokens=8000,
    )

    return {
        "warnings": warnings,
        "draft": draft,
        "rev_merit": rev_merit,
        "rev_proc": rev_proc,
        "final_text": final_formatted,
        "passages": passages,
        "attachments": anexos_nomes,
    }
