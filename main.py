import streamlit as st
from transformers import pipeline
from youtube_transcript_api import YouTubeTranscriptApi
from io import BytesIO
from gtts import gTTS
import re
import time

# Initialize the summarizer model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def validate_youtube_link(link):
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
    try:
        video_id = youtube_video_url.split("=")[1]
        with st.spinner("Fetching transcript... This might take a moment."):
            transcript_data = YouTubeTranscriptApi.get_transcript(video_id)
            transcript = " ".join([segment["text"] for segment in transcript_data])
            return transcript
    except Exception as e:
        st.error(f"Error fetching transcript: {e}")
        return None

def generate_summarization_with_huggingface_model(transcript_text, summary_ratio, chunk_size=500, overlap=50):
    try:
        total_text_length = len(transcript_text)
        summary_length = max(int(total_text_length * (summary_ratio / 100)), 50)  # Minimum length safeguard
        
        # Splitting transcript into overlapping chunks
        transcript_chunks = []
        start = 0
        while start < total_text_length:
            end = start + chunk_size
            transcript_chunks.append(transcript_text[start:end])
            start += chunk_size - overlap

        progress = st.progress(0)
        chunk_summaries = []
        total_chunks = len(transcript_chunks)
        
        # Adjust max summary length for each chunk
        max_chunk_summary_length = max(summary_length // total_chunks, 50)
        
        with st.spinner("Generating summary... This may take some time."):
            for i, chunk in enumerate(transcript_chunks):
                summary = summarizer(chunk, max_length=max_chunk_summary_length, min_length=20, do_sample=False)
                chunk_summaries.append(summary[0]["summary_text"])
                progress.progress((i + 1) / total_chunks)

        # Improved bullet formatting using regex to avoid breaking abbreviations
        sentences = re.split(r'(?<!\w\.\w)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', " ".join(chunk_summaries))
        bullet_summary = "\n\n".join([f"- {sentence.strip()}" for sentence in sentences if sentence])

        return bullet_summary
    except Exception as e:
        st.error(f"Error generating summary: {e}")
        return None

def text_to_speech(text, language='en'):
    tts = gTTS(text=text, lang=language, slow=False)
    fp = BytesIO()
    tts.write_to_fp(fp)
    return fp.getvalue()

def summarization():
    # st.title("Enter the YouTube video link:")
    # st.write("Summarize your YouTube video below:")
    st.caption("Enter the YouTube video link and select the summary percentage to generate a bullet-point summary.")
    youtube_link = st.text_input("Enter the YouTube video link here:")
    
    if youtube_link:
        video_id = validate_youtube_link(youtube_link)
        if video_id:
            st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", width=400)
    
    summary_ratio = st.slider("Select Summary Percentage", min_value=10, max_value=100, value=10, step=10)
    
    if st.button("Summarize"):
        if video_id:
            transcript_text = extract_transcript_details(youtube_link)
            if transcript_text:
                st.text("Summarizing the transcript...")
                summary = generate_summarization_with_huggingface_model(transcript_text, summary_ratio)
                if summary:
                    st.subheader("Summary:")
                    st.markdown(summary)
                    audio_data = text_to_speech(summary)
                    st.audio(audio_data, format="audio/mp3")
                else:
                    st.error("Failed to generate the summary. Please try again.")
        else:
            st.error("Invalid YouTube video link. Please check and try again.")

def main():
    st.title("YouTube Summarizer")
    st.write("Instantly summarize lengthy YouTube videos in bullet points.")
    st.divider()
    
    summarization()

if __name__ == "__main__":
    main()
