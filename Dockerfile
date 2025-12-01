FROM python:3.11-slim

# Não perguntar nada interativamente
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /app

# Instala dependências
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o código
COPY . .

# Comando de start: uvicorn na porta $PORT
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT}"]
