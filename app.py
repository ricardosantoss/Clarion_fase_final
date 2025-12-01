import os
import json
import uuid

import streamlit as st
from pydantic import ValidationError

from core import (
    StructuredInput,
    generate_petition,
    DEFAULT_PROVIDER,
    TOP_K,
)

st.set_page_config(page_title="DR.M — Petições com RAG (JSON)", layout="wide")

st.sidebar.title("Configurações")
provider = st.sidebar.selectbox(
    "Provider", ["sabia", "gpt"], index=0 if DEFAULT_PROVIDER == "sabia" else 1
)
with st.sidebar.expander("Parâmetros do RAG", expanded=False):
    top_k = st.slider("TOP_K (RAG)", 1, 12, int(TOP_K), 1)

st.header("DR.M — Petições com RAG (entrada JSON estruturada)")
st.markdown("Cole abaixo o JSON estruturado.")

example_json = """{ ... o mesmo exemplo grande que você já tinha ... }"""

json_input = st.text_area(
    "JSON do caso",
    value=example_json,
    height=500
)

go = st.button("Gerar Petição a partir do JSON")

if go:
    try:
        structured = StructuredInput.model_validate_json(json_input)
    except ValidationError as e:
        st.error("JSON inválido. Verifique o formato.")
        st.code(e.json(), language="json")
        st.stop()
    except Exception as e:
        st.error(f"Erro ao analisar JSON: {e}")
        st.stop()

    with st.spinner("Processando..."):
        result = generate_petition(
            structured,
            top_k=top_k,
            provider=provider,
        )

    for w in result["warnings"]:
        st.warning(w)

    st.success("Petições geradas! Veja abaixo.")

    with st.expander("Minuta (DRAFT)", expanded=True):
        st.write(result["draft"])

    colA, colB = st.columns(2)
    with colA:
        st.subheader("Revisor — Mérito")
        st.json(result["rev_merit"])
    with colB:
        st.subheader("Revisor — Procedimental")
        st.json(result["rev_proc"])

    st.subheader("Versão Final (Formatada)")
    st.write(result["final_text"])

    colD, colE = st.columns(2)
    with colD:
        st.download_button(
            "⬇️ Baixar DOCX",
            data=result["docx_bytes"],
            file_name=f"Peticao_Final_{uuid.uuid4().hex[:8]}.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
    with colE:
        st.download_button(
            "⬇️ Baixar PDF",
            data=result["pdf_bytes"],
            file_name=f"Peticao_Final_{uuid.uuid4().hex[:8]}.pdf",
            mime="application/pdf",
        )
