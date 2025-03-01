import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
import pickle
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = tf.keras.models.load_model("model.keras")

# Load or define the scaler (assuming it's saved as 'scaler.pkl')
try:
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    scaler = None
    st.warning("Scaler not found! Ensure you have the same scaler used during training.")

st.title("CNN Audio Classification App")
st.write("Upload a .wav file to classify if the speech is slurred or not.")

uploaded_file = st.file_uploader("Choose a WAV file", type=["wav"])

def preprocess_audio(file):
    y, sr = librosa.load(file, sr=None)  # Load audio file
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=128)  # Extract MFCC features
    mfccs = np.mean(mfccs, axis=1)  # Take mean across time axis
    return mfccs

if uploaded_file is not None:
    features = preprocess_audio(uploaded_file)
    features = features.reshape(1, -1)  # Reshape to match input size
    
    # Scale the features
    if scaler and hasattr(scaler, "mean_"):
        features_scaled = scaler.transform(features)
    else:
        st.error("Scaler is not fitted or not found. Ensure you use the correct scaler used during training.")
        features_scaled = features  # Fallback to unscaled
    
    features_scaled = features_scaled.reshape(1, 16, 8, 1)  # Reshape to CNN input shape
    prediction = model.predict(features_scaled)[0][0]
    
    st.write("Prediction Score:", prediction)
    
    if prediction > 0.5:
        st.success("The model predicts: Slurred Speech")
    else:
        st.success("The model predicts: Clear Speech")
