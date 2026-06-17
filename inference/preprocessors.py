import cv2
import librosa
import numpy as np
import torch
import torchvision.transforms as transforms
from transformers import DistilBertTokenizer
import sys
from pathlib import Path

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.config import MAX_LEN, TEXT_MODEL_NAME, N_MFCC, SAMPLE_RATE, DURATION, IMAGE_SIZE

class TextPreprocessor:
    def __init__(self):
        self.tokenizer = DistilBertTokenizer.from_pretrained(TEXT_MODEL_NAME)
        
    def preprocess(self, text):
        inputs = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=MAX_LEN,
            padding='max_length',
            return_token_type_ids=False,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return inputs['input_ids'], inputs['attention_mask']

class AudioPreprocessor:
    def preprocess(self, audio_data):
        """
        Preprocess audio data. Can accept a file path or a numpy array.
        """
        if isinstance(audio_data, str):
            y, sr = librosa.load(audio_data, sr=SAMPLE_RATE, duration=DURATION)
        else:
            # Assume it's a numpy array from microphone
            y = audio_data
        
        # Pad or truncate to fixed length
        length = SAMPLE_RATE * DURATION
        if len(y) < length:
            y = np.pad(y, (0, length - len(y)))
        else:
            y = y[:length]
            
        mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
        # Expected shape for CRNN: (1, 1, 40, time_steps)
        mfcc = torch.FloatTensor(mfcc).unsqueeze(0).unsqueeze(0)
        return mfcc

class VisionPreprocessor:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        # Use Haar cascade for fast, CPU-friendly face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
    def preprocess(self, frame):
        """
        Extract face from frame and apply transforms.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return None, None
            
        # Get the largest face (most likely the primary user)
        x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
        face_img = frame[y:y+h, x:x+w]
        
        # Convert BGR to RGB for MobileNet
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        tensor = self.transform(face_img).unsqueeze(0)
        
        return tensor, (x, y, w, h)
