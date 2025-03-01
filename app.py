import streamlit as st
from gtts import gTTS
import base64
import os
import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import scipy.io.wavfile as wav
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import io

# Function to convert text to speech
def text_to_speech(text):
    """Generate and return base64 audio string"""
    tts = gTTS(text=text, lang='en')
    filename = "output.mp3"
    tts.save(filename)

    with open(filename, "rb") as f:
        audio_bytes = f.read()

    return base64.b64encode(audio_bytes).decode()

# Load Wav2Vec2 model and processor from Hugging Face
def load_wav2vec2_model():
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")
    return processor, model

# Convert speech to text using wav2vec2
def speech_to_text(audio_input, processor, model):
    # Convert audio input to float32 before passing to the model
    audio_input = torch.tensor(audio_input, dtype=torch.float32)

    # Process the input and run through the model
    input_values = processor(audio_input, return_tensors="pt").input_values

    with torch.no_grad():
        logits = model(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
    return transcription

# Function to visualize audio frequency spectrum
def plot_frequency_spectrum(audio_input, sr=16000):
    plt.figure(figsize=(10, 6))
    plt.specgram(audio_input, NFFT=1024, Fs=2, noverlap=512)
    plt.title('Frequency Spectrum')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    st.pyplot(plt)

# Function to record audio from the microphone
def record_audio(duration=5, samplerate=16000):
    st.info(f"Recording for {duration} seconds...")
    audio_data = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()
    return audio_data.flatten()

def main():
    st.title("Text to Speech & Speech to Text Converter")

    # Tabs for navigation
    tab1, tab2 = st.tabs(["Text to Speech", "Speech to Text"])

    # Text to Speech Tab
    with tab1:
        text_input = st.text_area("Enter text:", "Hello, welcome to my Streamlit app!")
        
        if st.button("Convert to Speech"):
            if not text_input.strip():
                st.warning("Please enter some text to convert.")
            else:
                audio_base64 = text_to_speech(text_input)
                
                if audio_base64:
                    st.success("Conversion successful! Click play below.")
                    
                    # Display audio player
                    audio_html = f"""
                    <audio controls>
                        <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
                        Your browser does not support the audio element.
                    </audio>
                    """
                    st.markdown(audio_html, unsafe_allow_html=True)
                    
                    # Save the audio file for analysis
                    with open("output.mp3", "wb") as f:
                        f.write(base64.b64decode(audio_base64))
                    
                    # Visualize Frequency Spectrum
                    plot_frequency_spectrum(librosa.load("output.mp3", sr=16000)[0])
                    
                    # Load wav2vec2 model
                    processor, model = load_wav2vec2_model()

                    # Run speech-to-text
                    transcription = speech_to_text(librosa.load("output.mp3", sr=16000)[0], processor, model)
                    st.subheader("Transcription from Speech:")
                    st.write(transcription)

    # Speech to Text (Mic) Tab
    with tab2:
        st.info("Click the button below to start recording from your microphone.")
        
        if st.button("Start Recording"):
            audio_data = record_audio(duration=5)  # Record for 5 seconds
            
            st.success("Recording complete! Processing now...")
            
            # Visualize Frequency Spectrum of the recorded audio
            plot_frequency_spectrum(audio_data)
            
            # Load wav2vec2 model
            processor, model = load_wav2vec2_model()

            # Run speech-to-text
            transcription = speech_to_text(audio_data, processor, model)
            st.subheader("Transcription from Mic Input:")
            st.write(transcription)

            # Live Text Display
            st.text_area("Detected Text", value=transcription, height=100)

if __name__ == "__main__":
    main()
