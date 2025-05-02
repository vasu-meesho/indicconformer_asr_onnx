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
    allow_origins=["https://vasu-meesho.github.io"],  # Change to your domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)

model_dict= {
    "hi":"1UxQ4bL0-5QEQCiLI5tTu1A9IfI3SsJUO"
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
        #print("==========Input==========")
        #for i in self.model.get_inputs():
        #    print(i)
        #print("==========Output==========")
        #for i in self.model.get_outputs():
        #    print(i)
            
    def __call__(self, x: np.ndarray):
        # x: (T, C)
        x = torch.from_numpy(x)
        x = x.t().unsqueeze(0).unsqueeze(0)
        # x: [1, C, T]
        x_lens = torch.tensor([x.shape[-1]], dtype=torch.int64).unsqueeze(0)
        
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

# Ensure model exists
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, f"model_hi.onnx")
if not os.path.exists(MODEL_PATH):
    download_from_drive(model_dict[lang], MODEL_PATH)

# ----------------------------
# Load ONNX Runtime Session
# ----------------------------


id2token = dict()
with open(tokens, encoding="utf-8") as f:
    for line in f:
        t, idx = line.split()
        id2token[int(idx)] = t
blank = len(id2token) - 1


# ----------------------------
# FastAPI Setup
# ----------------------------



@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...), lang: str = Form(...)):
    model = OnnxModel(MODEL_PATH)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    audio, sample_rate = sf.read(tmp_path)
    #print(len(audio))
    if sample_rate != 16000:
        audio = librosa.resample(
            audio,
            orig_sr=sample_rate,
            target_sr=16000,
        )
        sample_rate = 16000
    fbank = create_fbank()
    features = compute_features(audio, fbank)
    features = torch.from_numpy(features)
    mean = features.mean(dim=1, keepdims=True)
    stddev = features.std(dim=1, keepdims=True) + 1e-5
    features = (features - mean) / stddev
    features = features.numpy()
    
    log_probs = model(features)
    os.remove(tmp_path)
    
    ans = []
    prev = -1
    log_probs = log_probs[0, :, :]  # remove batch dim
    #print(log_probs.shape)
    ids = torch.argmax(log_probs, dim=1).tolist()
    for k in ids:
        if k != blank and k != prev:
            ans.append(k)
        prev = k

    tokens = [id2token[i] for i in ans]
    underline = "▁"
    #  underline = b"\xe2\x96\x81".decode()
    text = "".join(tokens).replace(underline, " ").strip()
    return {"transcription": text}
