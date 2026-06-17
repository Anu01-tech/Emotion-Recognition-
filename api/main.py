from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import uvicorn
import numpy as np
import cv2
import io
import sys
from pathlib import Path

# Add parent directory to pythonpath
sys.path.append(str(Path(__file__).resolve().parent.parent))

from inference.engine import InferenceEngine

app = FastAPI(title="Multi-modal Emotion Recognition API", version="1.0.0")

# Initialize Engine (Loads models into memory)
try:
    engine = InferenceEngine(use_gpu=True)
except Exception as e:
    print(f"Error initializing Inference Engine: {e}")
    engine = None

class TextRequest(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Emotion Recognition API is running"}

@app.post("/predict/text")
def predict_text(request: TextRequest):
    if not engine:
        raise HTTPException(status_code=500, detail="Inference engine not initialized.")
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
        
    emotion, confidence, probs = engine.predict_text(request.text)
    return {
        "emotion": emotion,
        "confidence": confidence,
        "probabilities": probs
    }

@app.post("/predict/audio")
async def predict_audio(file: UploadFile = File(...)):
    if not engine:
        raise HTTPException(status_code=500, detail="Inference engine not initialized.")
        
    import soundfile as sf
    try:
        data, samplerate = sf.read(io.BytesIO(await file.read()))
        # Convert to mono if stereo
        if len(data.shape) > 1:
            data = data.mean(axis=1)
        emotion, confidence, probs = engine.predict_audio(data)
        return {
            "emotion": emotion,
            "confidence": confidence,
            "probabilities": probs
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

@app.post("/predict/vision")
async def predict_vision(file: UploadFile = File(...)):
    if not engine:
        raise HTTPException(status_code=500, detail="Inference engine not initialized.")
        
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
        
    emotion, confidence, probs, face_box = engine.predict_vision(frame)
    return {
        "emotion": emotion,
        "confidence": confidence,
        "probabilities": probs,
        "face_box": face_box
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
