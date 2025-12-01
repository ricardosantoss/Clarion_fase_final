from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core import StructuredInput, generate_petition

app = FastAPI(
    title="DR.M API",
    version="1.0.0",
)

# opcional: CORS liberado para o front dele
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # depois dá pra restringir
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/peticao")
def gerar_peticao(payload: StructuredInput):
    """
    Recebe o mesmo JSON que o Streamlit usa e devolve
    o pipeline multiagente (sem DOCX/PDF se não quiser).
    """
    result = generate_petition(payload)
    # se quiser pode remover docx_bytes/pdf_bytes da resposta
    return {
        "warnings": result["warnings"],
        "draft": result["draft"],
        "rev_merit": result["rev_merit"],
        "rev_proc": result["rev_proc"],
        "final_text": result["final_text"],
        "passages": result["passages"],
        "attachments": result["attachments"],
    }
