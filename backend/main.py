import os
import requests
import base64
import io
import kaldi_native_fbank as knf
import numpy as np
import soundfile as sf
import logging
import tempfile
import time

from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import onnxruntime as ort
import torch
import librosa
import gdown
from typing import Dict, Optional, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Indic ASR API",
    description="API for Indic language speech recognition using ONNX models",
    version="1.0.0"
)

# Configure CORS for deployment
# For production, replace with the specific domains you want to allow
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://vasu-meesho.github.io"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

# Model configuration
model_dict = {
    "hi": "1UxQ4bL0-5QEQCiLI5tTu1A9IfI3SsJUO"
}
MODEL_DIR = "models"
TOKENS_FILE = 'tokens.txt'

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Utility functions for downloading models
def download_from_gdrive(file_id: str, destination: str):
    """Download a file from Google Drive."""
    try:
        gdown.download(id=file_id, output=destination, quiet=False)
        return True
    except Exception as e:
        logger.error(f"Failed to download file: {e}")
        return False

def download_from_drive(file_id: str, destination: str):
    """Wrapper for Google Drive download with retry logic."""
    # Check if file already exists
    if os.path.exists(destination):
        logger.info(f"File already exists at {destination}")
        return True
    
    # Try to download with retries
    for attempt in range(3):
        try:
            logger.info(f"Downloading model (attempt {attempt+1}/3)...")
            success = download_from_gdrive(file_id, destination)
            if success:
                logger.info(f"Successfully downloaded to {destination}")
                return True
            time.sleep(2)  # Wait before retry
        except Exception as e:
            logger.warning(f"Download attempt {attempt+1} failed: {e}")
    
    logger.error("All download attempts failed")
    return False

def create_fbank():
    """Create the filterbank feature extractor."""
    opts = knf.FbankOptions()
    opts.frame_opts.dither = 0
    opts.frame_opts.samp_freq = 16000
    opts.frame_opts.window_type = "hanning"
    opts.frame_opts.frame_shift_ms = 10
    opts.frame_opts.frame_length_ms = 25
    opts.mel_opts.num_bins = 80
    opts.energy_floor = 0.0
    opts.mel_opts.debug_mel = False
    return knf.OnlineFbank(opts)

def compute_features(waveform: np.ndarray, fbank) -> np.ndarray:
    """Compute filterbank features from audio waveform."""
    waveform = waveform.astype(np.float32)
    fbank.accept_waveform(16000, waveform)
    frames = fbank.num_frames_ready
    mat = np.zeros([frames, 80])
    for i in range(frames):
        mat[i] = fbank.get_frame(i)
    fbank.reset()
    return mat

class OnnxModel:
    """Wrapper class for ONNX inference."""
    def __init__(self, filename: str):
        session_opts = ort.SessionOptions()
        session_opts.inter_op_num_threads = 1
        session_opts.intra_op_num_threads = 1
        self.session_opts = session_opts

        try:
            # Load the ONNX model
            self.model = ort.InferenceSession(
                filename,
                sess_options=self.session_opts,
                providers=["CPUExecutionProvider"],
            )
            
            # Log model inputs/outputs for debugging
            logger.info(f"Model loaded from {filename}")
            logger.info("Model inputs:")
            for i, inp in enumerate(self.model.get_inputs()):
                logger.info(f"  {i}: {inp.name} - shape: {inp.shape}")
            
            logger.info("Model outputs:")
            for i, out in enumerate(self.model.get_outputs()):
                logger.info(f"  {i}: {out.name}")
                
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Failed to load model: {e}")

    def __call__(self, x: np.ndarray):
        """Run inference on input tensor."""
        # x: (T, C) - Time x Features
        x = torch.from_numpy(x)
        x = x.t().unsqueeze(0).unsqueeze(0)  # Shape: [1, C, T]
        
        # Calculate sequence length
        x_lens = torch.tensor([x.shape[-1]], dtype=torch.int64).unsqueeze(0)
        
        logger.debug(f"Input shape: {x.shape}, Lens shape: {x_lens.shape}")
        
        try:
            # Run model inference
            log_probs = self.model.run(
                [self.model.get_outputs()[0].name],
                {
                    self.model.get_inputs()[0].name: x.numpy(),
                    self.model.get_inputs()[1].name: x_lens.numpy(),
                },
            )[0]
            return torch.from_numpy(log_probs)
            
        except Exception as e:
            logger.error(f"Inference error: {e}")
            raise RuntimeError(f"Inference failed: {e}")

# Caches for model and token data
model_cache: Dict[str, OnnxModel] = {}
id2token_cache: Dict[int, str] = {}

def get_model(lang: str) -> OnnxModel:
    """Get or initialize model for the specified language."""
    global model_cache
    
    if lang not in model_dict:
        raise ValueError(f"Language {lang} is not supported")
    
    if lang not in model_cache:
        model_path = os.path.join(MODEL_DIR, f"model_{lang}.onnx")
        
        # Check if model exists, download if needed
        if not os.path.exists(model_path):
            logger.info(f"Model for {lang} not found, downloading...")
            success = download_from_drive(model_dict[lang], model_path)
            if not success:
                raise RuntimeError(f"Failed to download model for language: {lang}")
        
        # Initialize model
        try:
            model_cache[lang] = OnnxModel(model_path)
        except Exception as e:
            logger.error(f"Error initializing model for {lang}: {e}")
            raise RuntimeError(f"Failed to initialize model: {e}")
    
    return model_cache[lang]

def get_id2token() -> Dict[int, str]:
    """Load token mapping from file."""
    global id2token_cache
    
    if not id2token_cache:
        id2token = {}
        try:
            with open(TOKENS_FILE, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) != 2:
                        logger.warning(f"Invalid token line: {line}")
                        continue
                        
                    t, idx = parts
                    id2token[int(idx)] = t
                    
            id2token_cache = id2token
            logger.info(f"Loaded {len(id2token)} tokens")
            
        except Exception as e:
            logger.error(f"Error loading tokens: {e}")
            raise RuntimeError(f"Failed to load tokens: {e}")
    
    return id2token_cache

# API endpoints
@app.get("/")
async def root():
    """Health check and API information endpoint."""
    return {
        "status": "online",
        "service": "Indic ASR API",
        "supported_languages": list(model_dict.keys()),
        "model_status": {lang: lang in model_cache for lang in model_dict}
    }

@app.get("/health")
async def health_check():
    """Health check endpoint required by some cloud platforms."""
    return {"status": "healthy"}

@app.post("/transcribe")
async def transcribe(request: Request, file: UploadFile = File(...), lang: str = Form(...)):
    """Transcribe audio file to text."""
    logger.info(f"New transcription request: lang={lang}, file={file.filename}")
    start_time = time.time()
    client_ip = request.client.host
    
    try:
        # Validate language
        if lang not in model_dict:
            return JSONResponse(
                status_code=400,
                content={"error": f"Language {lang} is not supported. Available languages: {list(model_dict.keys())}"}
            )
        
        # Get model for language
        try:
            model = get_model(lang)
        except Exception as e:
            logger.error(f"Error getting model: {e}")
            return JSONResponse(
                status_code=500, 
                content={"error": "Failed to load speech recognition model"}
            )
        
        # Save and process the uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            content = await file.read()
            file_size = len(content)
            logger.info(f"Received file size: {file_size/1024:.2f} KB")
            
            if file_size == 0:
                return JSONResponse(
                    status_code=400,
                    content={"error": "Empty audio file received"}
                )
                
            tmp.write(content)
            tmp_path = tmp.name

        try:
            # Read audio data
            try:
                audio, sample_rate = sf.read(tmp_path)
                logger.info(f"Audio loaded: length={len(audio)}, sample_rate={sample_rate}")
            except Exception as e:
                logger.error(f"Error reading audio file: {e}")
                return JSONResponse(
                    status_code=400,
                    content={"error": f"Invalid audio file: {str(e)}"}
                )
            
            # Resample if needed
            if sample_rate != 16000:
                logger.info(f"Resampling from {sample_rate} to 16000 Hz")
                audio = librosa.resample(
                    audio,
                    orig_sr=sample_rate,
                    target_sr=16000,
                )
                sample_rate = 16000
                
            # Convert to mono if needed
            if len(audio.shape) > 1 and audio.shape[1] > 1:
                logger.info(f"Converting from {audio.shape[1]} channels to mono")
                audio = librosa.to_mono(audio)
            
            # Feature extraction
            fbank = create_fbank()
            features = compute_features(audio, fbank)
            features = torch.from_numpy(features)
            
            # Normalize features
            mean = features.mean(dim=1, keepdim=True)
            stddev = features.std(dim=1, keepdim=True) + 1e-5
            features = (features - mean) / stddev
            features = features.numpy()
            
            logger.info(f"Feature shape: {features.shape}")
            
            # Run inference
            log_probs = model(features)
            
            # Decode results
            log_probs = log_probs[0, :, :]  # Remove batch dimension
            ids = torch.argmax(log_probs, dim=1).tolist()
            
            # Get token map
            id2token = get_id2token()
            
            # Apply CTC decoding (collapse repeated tokens and remove blanks)
            blank = len(id2token) - 1  # Assuming blank is the last token
            ans = []
            prev = -1
            for k in ids:
                if k != blank and k != prev:
                    ans.append(k)
                prev = k

            # Convert ids to text
            tokens = [id2token.get(i, "") for i in ans]
            underline = "▁"  # BPE separator token
            text = "".join(tokens).replace(underline, " ").strip()
            
            elapsed = time.time() - start_time
            logger.info(f"Transcription completed in {elapsed:.2f}s: '{text}'")
            
            return {"transcription": text, "processing_time": elapsed}
            
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"Error from {client_ip}: {error_details}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Server error: {str(e)}"}
        )

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
