import streamlit as st
from transformers import pipeline
import tempfile
import os
# Add the path to the 'bin' directory of FFmpeg to the system's PATH environment variable.
# This ensures that the Python program can locate and use the FFmpeg executable for audio and video processing.
# Replace 'C:\ffmpeg\bin' with the actual path where FFmpeg is installed if it differs.
os.environ["PATH"] += os.pathsep + r"C:\ffmpeg\bin"


# ------------------------------
# Load Whisper Model
# ------------------------------
def load_whisper_model():
    """
    Load the Whisper model for audio transcription.
    """
    transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")
    return transcriber

# ------------------------------
# Load NER Model
# ------------------------------
def load_ner_model():
    """
    Load the Named Entity Recognition (NER) model pipeline.
    """
    ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
    return ner_pipeline


# ------------------------------
# Transcription Logic
# ------------------------------


def transcribe_audio(uploaded_file):
    """
    Transcribe audio into text using the Whisper model.
    Args:
        uploaded_file: Audio file uploaded by the user.
    Returns:
        str: Transcribed text from the audio file.
    """
    # Save the uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(uploaded_file.read())
        temp_audio_path = temp_audio.name

    # Load Whisper model
    whisper_model = load_whisper_model()
    result = whisper_model(temp_audio_path, return_timestamps=True)

    return result['text']

# ------------------------------
# Entity Extraction
# ------------------------------
def extract_entities(text, ner_pipeline):
    """
    Extract entities from transcribed text using the NER model.
    Args:
        text (str): Transcribed text.
        ner_pipeline: NER pipeline loaded from Hugging Face.
    Returns:
        dict: Grouped entities (ORGs, LOCs, PERs).
    """
    entities = {'PER': set(), 'ORG': set(), 'LOC': set()}
    results = ner_pipeline(text)

    # Group entities
    for entity in results:
        group = entity['entity_group']
        if group in entities:
            entities[group].add(entity['word'])
    
    return entities

# ------------------------------
# Main Streamlit Application
# ------------------------------
def main():
    st.title("Meeting Transcription and Entity Extraction")

    # You must replace below
    STUDENT_NAME = "Burak Aydın "
    STUDENT_ID = "150220736"
    st.write(f"**{STUDENT_ID} - {STUDENT_NAME}**")

    # Upload audio file
    uploaded_file = st.file_uploader("Upload a WAV audio file", type=["wav"])

    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")

        # Transcription
        st.subheader("Transcription")
        transcribed_text = transcribe_audio(uploaded_file)
        st.write(transcribed_text)

        # Named Entity Extraction
        st.subheader("Named Entity Extraction")
        ner_pipeline = load_ner_model()
        entities = extract_entities(transcribed_text, ner_pipeline)

        # Display Entities
        st.subheader("Persons (PER):")
        if entities['PER']:
            for person in entities['PER']:
                st.write(f"- {person}")
        else:
            st.write("None")
        
        st.subheader("Organizations (ORG):")
        if entities['ORG']:
            for org in entities['ORG']:
                st.write(f"- {org}")
        else:
            st.write("None")
        
        st.subheader("Locations (LOC):")
        if entities['LOC']:
            for loc in entities['LOC']:
                st.write(f"- {loc}")
        else:
            st.write("None")


if __name__ == "__main__":
    main()
