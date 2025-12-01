from sentence_transformers import SentenceTransformer
import os

model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
print(f"Baixando modelo {model_name} para o cache do Docker...")
SentenceTransformer(model_name)
print("Modelo baixado com sucesso.")
