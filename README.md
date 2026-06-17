# Multi-modal Emotion Recognition System

A complete end-to-end multi-modal emotion recognition system covering Text, Speech, and Facial expressions.

## Project Structure
- `api/`: FastAPI backend for model inference.
- `app/`: Streamlit frontend dashboard.
- `models/`: PyTorch definitions for Vision (MobileNetV3), Audio (CRNN), Text (DistilBERT).
- `training/`: Training scripts and mock data loaders for the models.
- `inference/`: Real-time engine holding models in memory with optimized preprocessors.

## Setup

1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

2. **Download Data**
```bash
python data/download_scripts.py
```

3. **Train Models**
Run the training scripts to populate the `models/saved_models/` directory:
```bash
python training/train_text.py
python training/train_speech.py
python training/train_vision.py
```
*(Note: These scripts use dummy data for illustration out of the box, ready to be wired to the full datasets)*

## Running the Application

### 1. Start the FastAPI Backend
```bash
python api/main.py
```
Or via uvicorn directly:
```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Start the Streamlit Frontend
In a new terminal:
```bash
streamlit run app/app.py
```

## Docker Deployment

Build and run the containerized application:
```bash
docker build -t emotion-system .
docker run -p 8000:8000 -p 8501:8501 emotion-system
```
Access the dashboard at `http://localhost:8501`.
