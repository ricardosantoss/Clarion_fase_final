from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core import StructuredInput, generate_petition

app = FastAPI(
    title="DR.M API",
    version="1.0.0",
)

# CORS liberado (depois dá pra restringir)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/peticao")
def gerar_peticao(payload: StructuredInput):
    result = generate_petition(payload)
    return {
        "warnings": result["warnings"],
        "draft": result["draft"],
        "rev_merit": result["rev_merit"],
        "rev_proc": result["rev_proc"],
        "final_text": result["final_text"],
        "passages": result["passages"],
        "attachments": result["attachments"],
    }
