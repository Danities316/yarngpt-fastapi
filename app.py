import os
import time
import logging
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import Response
from pydantic import BaseModel
from typing import Optional
import torch
import torchaudio
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from outetts.wav_tokenizer.decoder import WavTokenizer
# from outetts.v0_1.decoder.pretrained import WavTokenizer  # Updated import
from yarngpt.audiotokenizer import AudioTokenizerV2

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_KEY = os.getenv("API_KEY", "1234567890")
WAV_TOKENIZER_CONFIG_PATH = os.getenv("WAV_TOKENIZER_CONFIG_PATH")
WAV_TOKENIZER_MODEL_PATH = os.getenv("WAV_TOKENIZER_MODEL_PATH")

# Lazy-loaded models
wav_tokenizer = None
model = None

def init_models():
    global wav_tokenizer, model
    if wav_tokenizer is None or model is None:
        start_time = time.time()
        logger.info("Initializing models with 8-bit quantization...")
        try:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16,
                bnb_8bit_use_double_quant=True
            )
            wav_tokenizer = WavTokenizer(
                config_path=WAV_TOKENIZER_CONFIG_PATH,
                model_path=WAV_TOKENIZER_MODEL_PATH,
                quantization_config=quantization_config
            )
            model = YarnGPT(quantization_config=quantization_config)
            logger.info(f"Model initialization took {time.time() - start_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Model initialization failed: {str(e)}")
            raise

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

class AudioRequest(BaseModel):
    text: str
    lang: str

@app.post("/generate-audio")
async def generate_audio(request: AudioRequest, x_api_key: Optional[str] = Header(None)):
    if x_api_key != API_KEY:
        logger.warning(f"Invalid API key: {x_api_key}")
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    start_time = time.time()
    try:
        init_models()  # Lazy load models
        logger.info(f"Processing request: text='{request.text}', lang='{request.lang}'")
        
        # Profile inference
        inference_start = time.time()
        audio = model.generate_audio(
            text=request.text,
            lang=request.lang,
            wav_tokenizer=wav_tokenizer
        )
        inference_time = time.time() - inference_start
        logger.info(f"Inference took {inference_time:.2f} seconds")
        
        total_time = time.time() - start_time
        logger.info(f"Total request processing took {total_time:.2f} seconds")
        return {"audio": audio.tolist()}
    except Exception as e:
        logger.error(f"Error generating audio: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating audio: {str(e)}")
