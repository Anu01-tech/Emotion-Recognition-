import streamlit as st
import requests
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import time

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Emotion Recognition Dashboard", layout="wide", page_icon="🎭")

# Setup CSS for rich aesthetics
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    h1 {
        color: #ff4b4b;
        font-family: 'Inter', sans-serif;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 5px;
        border: None;
        padding: 10px 24px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #ff7676;
    }
    .metric-card {
        background-color: #1e2127;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
</style>
""", unsafe_allow_html=True)

st.title("🎭 Multi-modal Emotion Recognition System")
st.markdown("Analyze emotions using Text, Audio, and Video.")

tab1, tab2, tab3 = st.tabs(["📝 Text", "🎤 Audio", "📷 Webcam Live"])

def plot_probs(probs, emotions):
    df = pd.DataFrame({
        'Emotion': emotions,
        'Probability': probs
    })
    fig = px.bar(df, x='Emotion', y='Probability', color='Emotion', 
                 title="Emotion Probabilities",
                 color_discrete_sequence=px.colors.qualitative.Pastel)
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

with tab1:
    st.header("Text Emotion Analysis")
    user_text = st.text_area("Enter your text here:", placeholder="I am feeling great today!")
    if st.button("Analyze Text"):
        if user_text:
            with st.spinner("Analyzing..."):
                try:
                    response = requests.post(f"{API_URL}/predict/text", json={"text": user_text})
                    if response.status_code == 200:
                        res = response.json()
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            st.markdown(f"### Predicted Emotion: **{res['emotion']}**")
                            st.progress(res['confidence'])
                            st.markdown(f"Confidence: {res['confidence']:.2%}")
                        with col2:
                            plot_probs(res['probabilities'], EMOTIONS)
                    else:
                        st.error("Error from API.")
                except Exception as e:
                    st.error(f"Could not connect to API. Is it running? Error: {e}")

with tab2:
    st.header("Audio Emotion Analysis")
    audio_file = st.file_uploader("Upload an audio file (WAV):", type=['wav'])
    if audio_file is not None:
        st.audio(audio_file)
        if st.button("Analyze Audio"):
            with st.spinner("Analyzing..."):
                try:
                    files = {"file": (audio_file.name, audio_file.getvalue(), "audio/wav")}
                    response = requests.post(f"{API_URL}/predict/audio", files=files)
                    if response.status_code == 200:
                        res = response.json()
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            st.markdown(f"### Predicted Emotion: **{res['emotion']}**")
                            st.progress(res['confidence'])
                            st.markdown(f"Confidence: {res['confidence']:.2%}")
                        with col2:
                            plot_probs(res['probabilities'], EMOTIONS)
                    else:
                        st.error("Error from API.")
                except Exception as e:
                    st.error(f"Could not connect to API. Is it running? Error: {e}")

with tab3:
    st.header("Real-time Facial Emotion Detection")
    st.markdown("We stream the webcam feed directly to the backend API for frame-by-frame analysis.")
    
    run = st.checkbox("Start Webcam")
    FRAME_WINDOW = st.image([])
    
    if run:
        cap = cv2.VideoCapture(0)
        
        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture video")
                break
                
            # Send frame to API
            _, buffer = cv2.imencode('.jpg', frame)
            files = {"file": ("frame.jpg", buffer.tobytes(), "image/jpeg")}
            
            try:
                response = requests.post(f"{API_URL}/predict/vision", files=files)
                if response.status_code == 200:
                    res = response.json()
                    emotion = res['emotion']
                    confidence = res['confidence']
                    box = res.get('face_box')
                    
                    if box:
                        x, y, w, h = box
                        # Draw bounding box
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(frame, f"{emotion} {confidence:.2f}", (x, y-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        
            except:
                cv2.putText(frame, "API Offline", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame)
            time.sleep(0.05) # Cap frame rate to not overwhelm API
            
        cap.release()
