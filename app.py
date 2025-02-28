import streamlit as st
from gtts import gTTS
import base64
import os

def text_to_speech(text):
    """Generate and return base64 audio string"""
    tts = gTTS(text=text, lang='en')
    filename = "output.mp3"
    tts.save(filename)

    with open(filename, "rb") as f:
        audio_bytes = f.read()

    return base64.b64encode(audio_bytes).decode()

def main():
    st.title("Text to Speech Converter")

    text_input = st.text_area("Enter text:", "Hello, welcome to my Streamlit app!")
    
    if st.button("Convert to Speech"):
        if not text_input.strip():
            st.warning("Please enter some text to convert.")
        else:
            audio_base64 = text_to_speech(text_input)
            
            if audio_base64:
                st.success("Conversion successful! Click play below.")
                
                audio_html = f"""
                <audio controls>
                    <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
                    Your browser does not support the audio element.
                </audio>
                """
                st.markdown(audio_html, unsafe_allow_html=True)
            else:
                st.error("Error: Unable to generate or access the audio file.")

if __name__ == "__main__":
    main()
