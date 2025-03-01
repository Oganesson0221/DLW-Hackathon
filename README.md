# Speech Processing App

## Overview
The **Speech Processing App** is a Streamlit-based web application designed to perform various speech-related tasks, including:

- **Text-to-Speech (TTS)** conversion using Google Text-to-Speech (gTTS).
- **Speech-to-Text (STT)** transcription using Wav2Vec2.
- **Sentiment Analysis** of transcribed text.
- **Emotion Detection** from speech.
- **Audio Frequency Spectrum Visualization**.
- **Real-time Microphone Recording and Processing**.
- **Slurred Speech Detection** using a CNN-based classifier.

## Features

### 1. Text-to-Speech (TTS)
- Converts input text to speech.
- Generates an audio file in MP3 format.
- Provides an embedded audio player to listen to the generated speech.
- Visualizes the frequency spectrum of the generated audio.

### 2. Speech-to-Text (STT)
- Uses **Wav2Vec2** to transcribe audio into text.
- Works with both uploaded audio files and real-time microphone input.
- Displays transcriptions along with frequency spectrum visualization.
### 3. Frequency Spectrum Analysis
- Demonstrates how audio signals are transformed into frequency spectrums.
- Provides visualization examples for different speech patterns.
    <img width="754" alt="Screenshot 2025-03-01 at 11 10 00 PM" src="https://github.com/user-attachments/assets/ac5465cb-a5b5-429f-a173-2394c1b01e3f" />


### 3. Sentiment Analysis
- Uses a **pre-trained sentiment analysis model** to classify transcribed text as positive, negative, or neutral.

### 4. Emotion Detection
- Predicts emotional tone from speech using an **emotion recognition model**.
- Displays the most probable emotion label along with a confidence score.
  <img width="810" alt="Screenshot 2025-03-01 at 11 18 30 PM" src="https://github.com/user-attachments/assets/7b4c316c-58c3-494d-8510-57ebe337fe03" />


### 5. Audio Classification (Slurred Speech Detection)
- Accepts uploaded `.wav` files for classification.
- Uses a **CNN-based deep learning model** trained to detect slurred speech with ~95% accuracy.
- Outputs whether the speech is clear or slurred based on model predictions.
  <img width="817" alt="Screenshot 2025-03-01 at 11 12 32 PM" src="https://github.com/user-attachments/assets/debe1c59-e9b9-45bc-9f79-3a1515e23701" />


## Jupyter Notebooks
We have included Jupyter notebooks to showcase:

1. **Speech Classification Model**
   - Achieves ~95% accuracy in detecting slurred speech.
   - Includes model training, evaluation, and performance metrics.


3. **Dataset Preparation and Labeling**
   - Explains the process of loading, preparing, and labeling the dataset.
   - Identifies high-quality audio samples for model training.

## How to Run the App

### Prerequisites
Make sure you have **Python 3.7+** installed along with the required dependencies.

### Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/your-repo/speech-processing-app.git
   cd speech-processing-app
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

### Running the App
To launch the Streamlit app, run:
```sh
streamlit run app.py
```

## Model Training & Dataset
- We first loaded, prepared, and labeled our dataset, identifying high-quality audio samples for model training.
- Subsequently, we harnessed **Wav2Vec2**, **CNN-based classifiers**, and **emotion detection models** to analyze speech patterns effectively.

## Future Enhancements
- Improve **real-time speech recognition** by optimizing microphone input processing.
- Expand **emotion recognition** with a broader dataset.
- Develop a **mobile-friendly UI** for better accessibility.

---
### Contributors
- Murugappan Venkatesh
- Rashi Ojha
- Rishika
- Shireen Verma
- Stanley Benjamin Yukon

For any issues or feature requests, please open an issue in the repository.

---
Thank you for using the Speech Processing App! 🚀

