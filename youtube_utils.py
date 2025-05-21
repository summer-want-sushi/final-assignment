from youtube_transcript_api import YouTubeTranscriptApi
import re

def get_youtube_video_id(query):
    try:
        match = re.search(r'(?:youtu\.be/|youtube\.com/(?:watch\?v=|embed/|v/|shorts/))([\w-]{11})', query)
        if match:
            video_id = match.group(1)
            print(video_id)
            return video_id
    except:
        print("Did not find youtube video id from query ", query)

def fetch_transcript_english(video_id):
    try:
        ytt_api = YouTubeTranscriptApi()
        transcript = ytt_api.fetch(video_id,languages=['en'])
        return transcript
    except:
        print("Error ")

def post_process_transcript(transcript_snippets):
    full_transcript = " ".join([transcript_snippet.text for transcript_snippet in transcript_snippets])
    return full_transcript
