import streamlit as st
from transformers import pipeline
from youtube_transcript_api import YouTubeTranscriptApi
import re
import time
from io import BytesIO
from gtts import gTTS  # Modified: Import gTTS for text-to-speech functionality

# Initialize the summarizer model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def validate_youtube_link(link):
    """
    Validates if the provided YouTube link is in the correct format.
    """
    youtube_regex = r"^(https?://)?(www\.)?(youtube\.com|youtu\.?be)/.+$"
    if re.match(youtube_regex, link):
        try:
            video_id = link.split("v=")[1].split("&")[0]
            return video_id
        except IndexError:
            st.error("Invalid YouTube link. Please check the format.")
            return None
    else:
        st.error("Invalid YouTube link. Please provide a valid URL.")
        return None


def extract_transcript_details(youtube_video_url):
    """
    Fetches the transcript for a YouTube video.
    """
    try:
        video_id = youtube_video_url.split("=")[1]
        with st.spinner("Fetching transcript... This might take a moment."):
            transcript_data = YouTubeTranscriptApi.get_transcript(video_id)
            transcript = " ".join([segment["text"] for segment in transcript_data])
            return transcript
    except Exception as e:
        st.error(f"Error fetching transcript: {e}")
        return None


def generate_summarization_with_huggingface_model(transcript_text, chunk_size=500, overlap=50):
    """
    Generate a summary using Hugging Face transformer model, handling long transcripts.
    """
    try:
        # Split transcript into smaller chunks
        transcript_chunks = []
        start = 0
        while start < len(transcript_text):
            end = start + chunk_size
            transcript_chunks.append(transcript_text[start:end])
            start += chunk_size - overlap  # Move by chunk size minus overlap

        # Summarize each chunk
        chunk_summaries = []
        with st.spinner("Generating summary... This may take some time."):
            for chunk in transcript_chunks:
                summary = summarizer(chunk, max_length=150, min_length=50, do_sample=False)
                chunk_summaries.append(summary[0]["summary_text"])

        # Combine all chunk summaries into a final summary
        final_summary = " ".join(chunk_summaries)
        return final_summary
    except Exception as e:
        st.error(f"Error generating summary: {e}")
        return None


def text_to_speech(text, language='en'):  # Modified: Added the text_to_speech function
    """
    Converts text to speech and returns the raw audio data.
    """
    tts = gTTS(text=text, lang=language, slow=False)
    fp = BytesIO()
    tts.write_to_fp(fp)
    return fp.getvalue()


def summarization():
    st.title("Summarization Page")
    st.write("Summarize your YouTube video below:")
    
    youtube_link = st.text_input("Enter the Youtube video link:")
    
    if youtube_link:
        video_id = validate_youtube_link(youtube_link)
        if video_id:
            st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_column_width=True)
    
    if st.button("Summarize"):
        if video_id:
            transcript_text = extract_transcript_details(youtube_link)
            if transcript_text:
                # Displaying a status message to indicate processing
                st.text("Summarizing the transcript...")
                summary = generate_summarization_with_huggingface_model(transcript_text)
                if summary:
                    st.subheader("Summary:")
                    st.write(summary)
                    audio_data = text_to_speech(summary)  # Modified: Convert the summary to audio
                    st.audio(audio_data, format="audio/mp3")  # Modified: Display the audio
                else:
                    st.error("Failed to generate the summary. Please try again.")
        else:
            st.error("Invalid YouTube video link. Please check and try again.")

def main():
    st.title("YouTube Summarizer")
    st.write("Instantly summarize lengthy YouTube videos, articles, and texts.")
    
    summarization()

if __name__ == "__main__":
    main()
