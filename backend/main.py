import os
import requests
import base64
import io
import kaldi_native_fbank as knf
import numpy as np
import soundfile as sf

import tempfile

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

import onnxruntime as ort
import torch
import librosa
import gdown
from utils import download_from_drive, download_from_gdrive, create_fbank, compute_features

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Changed to allow all origins for testing
    allow_methods=["*"],
    allow_headers=["*"],
)

model_dict = {
    "hi": "1UxQ4bL0-5QEQCiLI5tTu1A9IfI3SsJUO"
}
MODEL_DIR = "models"

tokens = 'tokens.txt'
    
class OnnxModel:
    def __init__(
        self,
        filename: str,
    ):
        session_opts = ort.SessionOptions()
        session_opts.inter_op_num_threads = 1
        session_opts.intra_op_num_threads = 1

        self.session_opts = session_opts

        self.model = ort.InferenceSession(
            filename,
            sess_options=self.session_opts,
            providers=["CPUExecutionProvider"],
        )
            
    def __call__(self, x: np.ndarray):
        # x: (T, C)
        x = torch.from_numpy(x)
        x = x.t().unsqueeze(0).unsqueeze(0)  # Changed: removed extra unsqueeze(0)
        # x: [1, C, T]

        x_lens = torch.tensor([x.shape[-1]], dtype=torch.int64).unsqueeze(0)  # Changed: removed unsqueeze(0)
        
        # Debug prints
        print(f"Input shape: {x.shape}")
        print(f"Input length shape: {x_lens.shape}")
        
        # Print model input requirements
        print("Model Input Requirements:")
        for inp in self.model.get_inputs():
            print(f"{inp.name}: shape={inp.shape}, type={inp.type}")
        
        # Fix inputs to match expected shapes
        log_probs = self.model.run(
            [
                self.model.get_outputs()[0].name,
            ],
            {
                self.model.get_inputs()[0].name: x.numpy(),
                self.model.get_inputs()[1].name: x_lens.numpy(),
            },
        )[0]
        # [batch_size, T, vocab_size]
        return torch.from_numpy(log_probs)


# Load models lazily
model_cache = {}

def get_model(lang):
    global model_cache
    
    if lang not in model_dict:
        raise ValueError(f"Language {lang} is not supported")
    
    if lang not in model_cache:
        # Ensure model exists
        os.makedirs(MODEL_DIR, exist_ok=True)
        MODEL_PATH = os.path.join(MODEL_DIR, f"model_{lang}.onnx")
        if not os.path.exists(MODEL_PATH):
            download_from_drive(model_dict[lang], MODEL_PATH)
        
        model_cache[lang] = OnnxModel(MODEL_PATH)
    
    return model_cache[lang]


# Load tokens only once
id2token_cache = {}

def get_id2token():
    global id2token_cache
    
    if not id2token_cache:
        id2token = {}
        try:
            with open(tokens, encoding="utf-8") as f:
                for line in f:
                    t, idx = line.strip().split()
                    id2token[int(idx)] = t
            id2token_cache = id2token
        except Exception as e:
            print(f"Error loading tokens: {e}")
            raise
    
    return id2token_cache


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...), lang: str = Form(...)):
    try:
        # Validate language
        if lang not in model_dict:
            return {"error": f"Language {lang} is not supported"}
        
        # Get model
        model = get_model(lang)
        
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            content = await file.read()
            print(f"Received audio file size: {len(content)} bytes")
            tmp.write(content)
            tmp_path = tmp.name

        try:
            # Process audio
            audio, sample_rate = sf.read(tmp_path)
            print(f"Audio loaded: length={len(audio)}, sample_rate={sample_rate}")
            
            if sample_rate != 16000:
                print(f"Resampling from {sample_rate} to 16000")
                audio = librosa.resample(
                    audio,
                    orig_sr=sample_rate,
                    target_sr=16000,
                )
                sample_rate = 16000
                
            # Create features
            fbank = create_fbank()
            features = compute_features(audio, fbank)
            features = torch.from_numpy(features)
            
            # Normalize features
            mean = features.mean(dim=1, keepdim=True)
            stddev = features.std(dim=1, keepdim=True) + 1e-5
            features = (features - mean) / stddev
            features = features.numpy()
            
            print(f"Feature shape: {features.shape}")
            
            # Run inference
            log_probs = model(features)
            
            # Decode results
            log_probs = log_probs[0, :, :]  # remove batch dim
            ids = torch.argmax(log_probs, dim=1).tolist()
            
            # Get token map
            id2token = get_id2token()
            blank = len(id2token) - 1
            
            # Apply CTC decoding
            ans = []
            prev = -1
            for k in ids:
                if k != blank and k != prev:
                    ans.append(k)
                prev = k

            # Convert to text
            tokens = [id2token[i] for i in ans if i in id2token]
            underline = "▁"
            text = "".join(tokens).replace(underline, " ").strip()
            
            return {"transcription": text}
            
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in transcription: {error_details}")
        return {"error": str(e), "details": error_details}


# Add a simple health check endpoint
@app.get("/")
async def root():
    return {"status": "running", "supported_languages": list(model_dict.keys())}
