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
-------------
# Speech Processing App

## Overview
This Streamlit app provides a comprehensive speech processing pipeline with features including:
- **Text-to-Speech Conversion (TTS)**
- **Speech-to-Text Conversion (STT)**
- **Audio Classification for Slurred Speech Detection**
- **Sentiment Analysis and Emotion Detection**
- **Audio Frequency Spectrum Visualization**

## Features
### 1. Text-to-Speech (TTS)
- Users can input text, which is then converted to speech using `gTTS`.
- The generated speech can be played directly in the app.
- **Additional Features:**
  - Visual representation of the frequency spectrum.
  - Speech-to-text conversion of the generated audio.
  - Sentiment analysis and emotion detection on the transcribed text.

### 2. Speech-to-Text (STT)
- Users can record their speech using the microphone.
- The recorded speech is transcribed using `Wav2Vec2`.
- **Additional Features:**
  - Sentiment analysis and emotion detection on the transcribed text.
  - Visualization of the audio's frequency spectrum.

### 3. Audio Classification (Slurred Speech Detection)
- Users can upload a `.wav` file for classification.
- A trained CNN model predicts whether the speech is slurred or clear.
- **Model Accuracy:** The model achieves **95% accuracy** based on our dataset.

### 4. Sentiment Analysis & Emotion Detection
- The transcribed text from STT is analyzed for sentiment and emotions.
- Pre-trained transformers (`distilroberta-base`) are used for accurate predictions.

### 5. Frequency Spectrum Visualization
- The app provides a visualization of the frequency spectrum of audio signals.
- *Placeholder for Example Image:*
  
  ![Example Frequency Spectrum](path/to/your/image.png)  
  *(Replace the path with the actual image file)*

## Datasets & Model Training
We followed a structured approach to ensure high model accuracy:
1. **Dataset Preparation:**
   - We curated and labeled a high-quality dataset of speech samples.
   - Data cleaning and preprocessing were performed to remove noise.
2. **Data Augmentation:**
   - We employed two techniques: `MixupGenerator` and `RandomEraser` to enhance model generalization.
3. **Model Training:**
   - We trained a CNN-based model to classify speech as either **Slurred Speech** or **Clear Speech**.
   - Achieved a classification accuracy of **95%** on the test dataset.


## Dependencies
The project requires the following libraries:
- `streamlit`
- `gtts`
- `torch`
- `librosa`
- `numpy`
- `matplotlib`
- `sounddevice`
- `transformers`
- `tensorflow`
- `scikit-learn`

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
- [Murugappan Venkatesh](https://github.com/Murugapz)
- [Rashi Ojha](https://github.com/Rashi0192)
- [Rishika Mehta](https://github.com/Oganesson0221)
- [Shireen Verma](https://github.com/s812v)
- [Stanley Benjamin Yukon](https://github.com/WinAmazingPrizes)

For any issues or feature requests, please open an issue in the repository.

---
Thank you for using the Speech Processing App! 🚀

