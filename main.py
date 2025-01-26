import streamlit as st
from io import BytesIO
from gtts import gTTS
from gensim.summarization import summarize
import sqlite3  # For database interaction
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline

# Constants and model
summarizer = pipeline("summarization")

def text_to_speech(text, language='en'):
    """
    Converts text to speech and returns the raw audio data.
    """
    tts = gTTS(text=text, lang=language, slow=False)
    fp = BytesIO()
    tts.write_to_fp(fp)
    return fp.getvalue()


def generate_summary(text, ratio=0.2):
    """
    Generates a summary of the provided text.
    """
    try:
        summary = summarize(text, ratio=ratio)
        return summary
    except Exception as e:
        st.error(f"Error generating summary: {e}")
        return None


def extract_transcript_details(youtube_video_url):
    try:
        video_id = youtube_video_url.split("=")[1]
        transcript_text = YouTubeTranscriptApi.get_transcript(video_id)

        transcript = " ".join(segment["text"] for segment in transcript_text)
        return transcript
    except Exception as e:
        st.error(f"Error fetching transcript: {e}")
        return None


def generate_summarization_with_huggingface_model(transcript_text):
    """
    Generate summary using Hugging Face transformer model.
    """
    try:
        summary = summarizer(transcript_text, max_length=150, min_length=50, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        st.error(f"Error generating summary: {e}")
        return None


def summarization():
    st.title("Summarization Page")
    st.write("Summarize your YouTube video below:")
    youtube_link = st.text_input("Enter the Youtube video link:")

    if youtube_link:
        video_id = youtube_link.split("=")[1]
        st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_column_width=1200)

    if st.button("**Summarize**"):
        transcript_text = extract_transcript_details(youtube_link)
        if transcript_text:
            summary = generate_summarization_with_huggingface_model(transcript_text)
            st.subheader("Summary:")
            st.write(summary)
            st.session_state["summary_text"] = summary  # Store summary for potential audio generation
            
            # Text-to-speech conversion
            audio_data = text_to_speech(summary)
            st.audio(audio_data, format="audio/mp3")

            
        else:
            st.error("Error: Unable to fetch transcript. Please try again.")


def main():
    st.title("YouTube Summarizer")
    st.write("Instantly summarize lengthy YouTube videos, articles, and texts.")

    summarization()


if __name__ == "__main__":
    main()
