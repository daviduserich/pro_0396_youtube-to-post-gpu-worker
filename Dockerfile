# 1. Basis: Python 3.10 Slim
FROM python:3.10-slim

# --- NEU: Flexible Steuerung für GitHub Actions ---
ARG MODEL_ID="openai/whisper-large-v3"
ARG LANG="auto"

ENV WHISPER_MODEL=$MODEL_ID
ENV WHISPER_LANG=$LANG
# Fragmentierung verhindern (WICHTIG für lange Files!)
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# --------------------------------------------------

# 2. System-Tools
ENV DEBIAN_FRONTEND=noninteractive

# UPDATE: "procps" hinzugefügt (wichtig für psutil/RAM-Stats), ffmpeg für Audio-Cleaning
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    build-essential \
    procps \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 user

# 3. Workdir
WORKDIR /home/user/app

# 4. Pip Upgrade
RUN pip install --no-cache-dir --upgrade pip

# 5. Schritt A: PyTorch
RUN pip install --no-cache-dir \
    torch==2.4.0 \
    torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu121

# 6. Schritt B: NumPy Fix
RUN pip install --no-cache-dir "numpy<2.0"

# 7. Schritt C: Hugging Face & Pyannote
RUN pip install --no-cache-dir \
    "huggingface_hub==0.23.0" \
    "pyannote.audio==3.3.1"

# 8. Schritt D: Webserver & Helper
RUN pip install --no-cache-dir \
    "transformers==4.46.1" \
    "fastapi" \
    "uvicorn" \
    "python-multipart" \
    "accelerate" \
    "scipy" \
    "matplotlib" \
    "psutil" \
    "librosa"

# 9. App Code kopieren
COPY --chown=user app.py /home/user/app/app.py
COPY --chown=user worker.py /home/user/app/worker.py

# 10. Start
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
