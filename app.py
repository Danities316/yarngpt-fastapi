import os
import logging
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import Response
from pydantic import BaseModel
import torch
import torchaudio
from transformers import AutoModelForCausalLM, AutoTokenizer
from outetts.wav_tokenizer.decoder import WavTokenizer
# from outetts.v0_1.decoder.pretrained import WavTokenizer  # Updated import
from yarngpt.audiotokenizer import AudioTokenizerV2
 
 # Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
 
 # Initialize FastAPI app
app = FastAPI()
 
 # Environment variables
API_KEY = os.getenv("API_KEY", "secret-key")
logger.info(f"Expected API_KEY: {API_KEY}")
TOKENIZER_PATH = "saheedniyi/YarnGPT2"
 
WAV_TOKENIZER_CONFIG_PATH = os.getenv("WAV_TOKENIZER_CONFIG_PATH", "./wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml")
WAV_TOKENIZER_MODEL_PATH = os.getenv("WAV_TOKENIZER_MODEL_PATH", "./wavtokenizer_large_speech_320_24k.ckpt")
 
# Supported languages and speakers
SUPPORTED_LANGUAGES = {
    "ha": {"lang": "hausa", "speaker": "hausa_female1"},
    "yo": {"lang": "yoruba", "speaker": "yoruba_male1"},
    "ig": {"lang": "igbo", "speaker": "igbo_female1"},
    "en": {"lang": "english", "speaker": "pidgin_male1"}
}
 
# Force CPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = ""
torch.set_num_threads(2)  # For Codespaces 2-core
 
# Custom AudioTokenizerV2 wrapper to enforce weights_only=True
class SafeAudioTokenizerV2(AudioTokenizerV2):
    def __init__(self, tokenizer_path, wav_tokenizer_model_path, wav_tokenizer_config_path):
        try:
            # Validate model file before loading
            if not os.path.exists(wav_tokenizer_model_path):
                raise FileNotFoundError(f"WavTokenizer model file not found: {wav_tokenizer_model_path}")
            with open(wav_tokenizer_model_path, "rb") as f:
                if f.read(2) == b"<!":
                    raise ValueError(f"WavTokenizer model file appears to be an HTML page: {wav_tokenizer_model_path}")
            super().__init__(tokenizer_path, wav_tokenizer_model_path, wav_tokenizer_config_path)
        except Exception as e:
            raise Exception(f"Failed to initialize AudioTokenizerV2: {str(e)}")
 
# Load model and tokenizer at startup
logger.info("Loading YarnGPT model and tokenizer...")
try:
    audio_tokenizer = SafeAudioTokenizerV2(
        TOKENIZER_PATH, WAV_TOKENIZER_MODEL_PATH, WAV_TOKENIZER_CONFIG_PATH
    )
    model = AutoModelForCausalLM.from_pretrained(
        TOKENIZER_PATH, torch_dtype=torch.float32, device_map="cpu"
    ).to("cpu")
    logger.info("Model and tokenizer loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise Exception(f"Model initialization failed: {str(e)}")
 
# Request model
class AudioRequest(BaseModel):
    text: str
    lang: str
 
@app.post("/generate-audio")
async def generate_audio(request: AudioRequest, x_api_key: str = Header(...)):
    """
    Generate Nigerian-accented audio from text.
    Input: { "text": string, "lang": string ("ha", "yo", "ig", "en") }
    Output: WAV audio file as binary response
    """
    if x_api_key != API_KEY:
        error_msg = f"Invalid API key: {x_api_key} does not match expected key"
        logger.error(error_msg)
        raise HTTPException(status_code=401, detail=error_msg)
 
    if not request.text or len(request.text.strip()) == 0:
        logger.error("Empty text provided")
        raise HTTPException(status_code=400, detail="Text cannot be empty")
 
    if request.lang not in SUPPORTED_LANGUAGES:
        error_msg = f"Unsupported language: {request.lang}. Use one of: {', '.join(SUPPORTED_LANGUAGES.keys())}"
        logger.error(error_msg)
        raise HTTPException(status_code=400, detail=error_msg)
 
    try:
        logger.info(f"Generating audio for text: {request.text}, lang: {request.lang}")
        lang_info = SUPPORTED_LANGUAGES[request.lang]
 
        prompt = audio_tokenizer.create_prompt(
            request.text,
            lang=lang_info["lang"],
            speaker_name=lang_info["speaker"]
        )
        input_ids = audio_tokenizer.tokenize_prompt(prompt)
 
        output = model.generate(
            input_ids=input_ids,
            temperature=0.1,
            repetition_penalty=1.1,
            max_length=2000,
            do_sample=True
        )
 
        codes = audio_tokenizer.get_codes(output)
        audio = audio_tokenizer.get_audio(codes)
 
        temp_file = "/tmp/output.wav"
        torchaudio.save(temp_file, audio, sample_rate=24000)
 
        with open(temp_file, "rb") as f:
            audio_data = f.read()
 
        os.remove(temp_file)
 
        logger.info("Audio generated successfully")
        return Response(
            content=audio_data,
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=output.wav"}
        )
 
    except Exception as e:
        logger.error(f"Error generating audio: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate audio: {str(e)}")
 
@app.get("/health")
async def health_check():
    return {"status": "it is healthy"}
