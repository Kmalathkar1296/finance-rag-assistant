# ---------- Base image ----------
    FROM python:3.10-slim

    # ---------- Environment settings ----------
    ENV PYTHONDONTWRITEBYTECODE=1 \
        PYTHONUNBUFFERED=1 \
        PIP_NO_CACHE_DIR=1
    
    # ---------- System dependencies ----------
    RUN apt-get update && apt-get install -y \
        build-essential \
        git \
        curl \
        && rm -rf /var/lib/apt/lists/*
    
    # ---------- Set working directory ----------
    WORKDIR /app
    
    # ---------- Copy dependency files first (for layer caching) ----------
    COPY requirements.txt .
    COPY setup.py .
    
    # ---------- Install Python dependencies ----------
    RUN pip install --upgrade pip \
        && pip install -r requirements.txt
    
    # ---------- Copy project source ----------
    COPY src/ src/
    COPY scripts/ scripts/
    COPY notebooks/ notebooks/
    COPY tests/ tests/
    COPY data/ data/
    COPY outputs/ outputs/
    COPY chroma_db/ chroma_db/
    COPY README.md .
    COPY .env.example .
    COPY .gitignore .
    
    # ---------- Make src importable ----------
    ENV PYTHONPATH=/app
    
    # ---------- Expose port (if using Gradio / API) ----------
    EXPOSE 7860
    
    # ---------- Default command ----------
    # Change this depending on how you run the app
    CMD ["python", "app.py"]
    