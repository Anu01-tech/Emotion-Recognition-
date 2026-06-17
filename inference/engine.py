import torch
import torch.nn.functional as F
import sys
from pathlib import Path
from transformers import pipeline

# Add parent dir to sys path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.speech_model import SpeechEmotionModel
from models.vision_model import VisionEmotionModel
from inference.preprocessors import AudioPreprocessor, VisionPreprocessor
from utils.config import NUM_CLASSES, EMOTIONS, MODEL_DIR

class InferenceEngine:
    """
    Singleton-like manager to hold all 3 models in memory for fast real-time inference.
    """
    def __init__(self, use_gpu=True):
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        print(f"Initializing InferenceEngine on device: {self.device}")
        
        # Initialize Models
        self.speech_model = SpeechEmotionModel(NUM_CLASSES).to(self.device)
        self.vision_model = VisionEmotionModel(NUM_CLASSES).to(self.device)
        
        print("Loading HuggingFace text pipeline...")
        self.text_pipeline = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None,
            device=0 if torch.cuda.is_available() and use_gpu else -1
        )
        
        self.load_weights()
        
        # Set to evaluation mode
        self.speech_model.eval()
        self.vision_model.eval()
        
        # Initialize Preprocessors
        self.audio_prep = AudioPreprocessor()
        self.vision_prep = VisionPreprocessor()
        
    def load_weights(self):
        speech_path = MODEL_DIR / "speech_model.pth"
        vision_path = MODEL_DIR / "vision_model.pth"
            
        if speech_path.exists():
            self.speech_model.load_state_dict(torch.load(speech_path, map_location=self.device))
        else:
            print("Warning: Speech model weights not found. Using randomly initialized weights.")
            
        if vision_path.exists():
            self.vision_model.load_state_dict(torch.load(vision_path, map_location=self.device))
        else:
            print("Warning: Vision model weights not found. Using randomly initialized weights.")

    def _predict(self, model, *args):
        with torch.no_grad():
            outputs = model(*args)
            probs = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)
        return EMOTIONS[predicted.item()], confidence.item(), probs[0].cpu().numpy().tolist()

    def predict_text(self, text):
        results = self.text_pipeline(text)[0]
        # Map labels to our emotions
        mapping = {
            "anger": "Angry", "disgust": "Disgust", "fear": "Fear", 
            "joy": "Happy", "sadness": "Sad", "surprise": "Surprise", "neutral": "Neutral"
        }
        
        # Find highest score
        top_res = max(results, key=lambda x: x['score'])
        predicted_emotion = mapping.get(top_res['label'], "Neutral")
        confidence = top_res['score']
        
        # Build probs array in the order of EMOTIONS
        probs = [0.0] * NUM_CLASSES
        for res in results:
            idx = EMOTIONS.index(mapping.get(res['label'], "Neutral"))
            probs[idx] = res['score']
            
        return predicted_emotion, confidence, probs
        
    def predict_audio(self, audio_data):
        mfcc = self.audio_prep.preprocess(audio_data)
        return self._predict(self.speech_model, mfcc.to(self.device))
        
    def predict_vision(self, frame):
        tensor, face_box = self.vision_prep.preprocess(frame)
        if tensor is None:
            return "No Face", 0.0, [0]*NUM_CLASSES, None
            
        emotion, confidence, probs = self._predict(self.vision_model, tensor.to(self.device))
        return emotion, confidence, probs, face_box
