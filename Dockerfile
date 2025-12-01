FROM python:3.11-slim

# Variáveis de ambiente para evitar arquivos .pyc e logs em buffer
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000 \
    # Define o cache do HuggingFace para dentro da pasta app para persistir se necessário
    HF_HOME=/app/cache

WORKDIR /app

# Instala dependências do sistema necessárias para compilar algumas libs python
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Instala as dependências (nota: o torch cpu já está no requirements com a url correta)
RUN pip install --no-cache-dir -r requirements.txt

# Copia o script de preload
COPY preload_model.py .
# Executa o download do modelo AGORA (build time), não no runtime
RUN python preload_model.py

COPY . .

# O Render injeta a variável PORT automaticamente, mas o padrão é 10000.
# O shell formata corretamente.
CMD uvicorn api:app --host 0.0.0.0 --port $PORT
