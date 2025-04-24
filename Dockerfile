# Use Python 3.10 slim image for compatibility with outetts>=0.2.3
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies, including build tools for pesq
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    wget \
    gcc \
    libc-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    --index-url https://pypi.org/simple \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Clean up build dependencies to reduce image size
RUN apt-get purge -y --auto-remove gcc libc-dev \
    && apt-get clean

# Clone and install the yarngpts submodule
RUN git clone https://github.com/saheedniyi02/yarngpt.git /app/yarngpt

# Download WavTokenizer config and model files
RUN !wget https://huggingface.co/novateur/WavTokenizer-medium-speech-75token/resolve/main/wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml
RUN !gdown 1-ASeEkrn4HY49yZWHTASgfGFNXdVnLTt

# Copy application code
COPY app.py .
COPY .gitmodules .
COPY yarngpts yarngpts

# Expose port for FastAPI
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=""
ENV WAV_TOKENIZER_CONFIG_PATH=/app/wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml
ENV WAV_TOKENIZER_MODEL_PATH=/app/wavtokenizer_large_speech_320_24k.ckpt

# Run FastAPI with Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]