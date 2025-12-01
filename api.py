import logging
import traceback
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from core import StructuredInput, generate_petition

# Configura logs para aparecerem no painel do Render
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="DR.M API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def healthcheck():
    return {"status": "ok", "message": "DR.M API rodando"}

@app.post("/peticao")
def gerar_peticao(payload: StructuredInput):
    try:
        logger.info(f"Recebendo pedido para caso: {payload.case_data.forum}")
        
        # Chama a função core
        result = generate_petition(payload)
        
        logger.info("Petição gerada com sucesso.")
        return {
            "warnings": result["warnings"],
            "draft": result["draft"],
            "rev_merit": result["rev_merit"],
            "rev_proc": result["rev_proc"],
            "final_text": result["final_text"],
            "passages": result["passages"],
            "attachments": result["attachments"],
        }
    except Exception as e:
        # Isso vai imprimir o erro completo no painel do Render
        error_msg = traceback.format_exc()
        logger.error(f"Erro ao gerar petição: {error_msg}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

