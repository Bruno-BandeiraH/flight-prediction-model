# ===============================
# Base image
# ===============================
FROM python:3.12-slim

# ===============================
# Variáveis de ambiente
# ===============================
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ===============================
# Diretório de trabalho
# ===============================
WORKDIR /app

# ===============================
# Dependências do sistema
# (necessárias para numpy, scipy, xgboost)
# ===============================
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ===============================
# Copia requirements primeiro
# (melhora cache de build)
# ===============================
COPY requirements.txt .

# ===============================
# Instala dependências Python
# ===============================
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ===============================
# Copia o restante da aplicação
# ===============================
COPY . .

# ===============================
# Porta exposta (documentação)
# ===============================
EXPOSE 8000

# ===============================
# Comando de inicialização
# ===============================
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
