
+# DR.M API
+
+API em FastAPI voltada à geração automática de petições jurídicas no contexto brasileiro. A aplicação recebe dados estruturados do caso, processa documentos anexados com RAG (FAISS + BM25) e utiliza provedores LLM configuráveis para redigir, revisar e formatar a petição final.
+
+## Arquitetura e fluxo
+1. **Recepção dos dados**: o endpoint `/peticao` recebe `StructuredInput` contendo dados do caso, configurações dos agentes e anexos em PDF codificados em base64.
+2. **Ingestão e busca (RAG)**: os PDFs são lidos (`pypdf`/`pdfplumber`), transformados em chunks e indexados com embeddings (`sentence-transformers`) no FAISS. Opcionalmente o BM25 é usado para fusão dos resultados.
+3. **Geração da minuta**: um agente escritor produz a primeira minuta com base no resumo do caso e trechos recuperados.
+4. **Revisões**: agentes específicos avaliam mérito e procedimento, retornando apontamentos em JSON.
+5. **Formatação final**: um agente de formatação aplica as correções e retorna o texto final estruturado.
+
+## Endpoints
+- `GET /` — Health check simples.
+- `POST /peticao` — Gera a petição a partir de um `StructuredInput`.
+
+### Estrutura do payload (resumo)
+```json
+{
+  "case_data": {
+    "forum": "Comarca X",
+    "parties": [
+      {"role": "plaintiff", "name": "Autor", "person_type": "PF"},
+      {"role": "defendant", "name": "Réu", "person_type": "PJ"}
+    ],
+    "facts_summary": "Resumo dos fatos...",
+    "claims": [
+      {"title": "Obrigação de fazer", "type": "principal", "details": "Detalhes"}
+    ],
+    "case_value": {"currency": "BRL", "value": 10000},
+    "urgency": false
+  },
+  "agents": {
+    "writer": {"temperature": 0.4},
+    "reviewer_merit": {"temperature": 0.4},
+    "reviewer_proc": {"temperature": 0.4},
+    "reviewer_format": {"temperature": 0.2}
+  },
+  "attachments": [
+    {"filename": "anexo.pdf", "content": "<base64-do-pdf>"}
+  ]
+}
+```
+
+## Variáveis de ambiente principais
+- `LLM_PROVIDER` (padrão: `sabia`) — Define o provedor (`sabia` ou `gpt`).
+- `OPENAI_API_KEY` — Necessária quando `LLM_PROVIDER=gpt`.
+- `SABIA_API_KEY` e `SABIA_BASE_URL` — Necessárias quando `LLM_PROVIDER=sabia`.
+- `MODEL_GPT` / `MODEL_SABIA` — Modelos usados por provedor.
+- `EMBEDDING_MODEL` — Modelo de embeddings para FAISS (padrão `sentence-transformers/all-MiniLM-L6-v2`).
+- `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`, `USE_BM25` — Ajustes de chunking e busca.
+- `PORT` — Porta utilizada pelo servidor Uvicorn (padrão 8000 ou definido pelo ambiente de deploy).
+
+## Execução local
+### Requisitos
+- Python 3.11
+- Dependências listadas em `requirements.txt` (inclui `torch` CPU e `faiss-cpu`).
+
+### Passos
+1. Crie e ative um ambiente virtual.
+2. Instale as dependências:
+   ```bash
+   pip install -r requirements.txt
+   ```
+3. Configure as variáveis de ambiente necessárias.
+4. Inicie o servidor:
+   ```bash
+   uvicorn api:app --host 0.0.0.0 --port 8000
+   ```
+5. Acesse `http://localhost:8000/docs` para testar via Swagger UI.
+
+## Execução via Docker
+O `Dockerfile` faz o pré-download do modelo de embeddings durante o build para acelerar o start em produção.
+
+```bash
+docker build -t dr-m-api .
+docker run -p 8000:8000 \
+  -e LLM_PROVIDER=sabia \
+  -e SABIA_API_KEY=... \
+  -e SABIA_BASE_URL=... \
+  dr-m-api
+```
+
+## Estrutura dos arquivos principais
+- `api.py` — Define o app FastAPI, middleware CORS e endpoints.
+- `core.py` — Modelos Pydantic, utilidades de ingestão de PDFs, indexador vectorial, prompts e fluxo de geração/revisão da petição.
+- `preload_model.py` — Baixa o modelo de embeddings durante o build Docker.
+- `requirements.txt` — Lista de dependências do projeto.
+
+## Observações
+- O OCR não está habilitado; PDFs sem texto retornam aviso em `warnings`.
+- Para usar provedores LLM é obrigatório configurar as chaves correspondentes.
