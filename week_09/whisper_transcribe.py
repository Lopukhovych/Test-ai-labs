# whisper_transcribe.py
from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

def transcribe(audio_path: str, language: str = None) -> str:
    """Transcribe audio file to text using Whisper."""
    with open(audio_path, "rb") as f:
        params = {
            "model": "whisper-1",
            "file": f,
            "response_format": "text"
        }
        if language:
            params["language"] = language  # e.g. "en", "fr", "de"

        return client.audio.transcriptions.create(**params)

def transcribe_with_timestamps(audio_path: str) -> dict:
    """Transcribe with word-level timestamps."""
    with open(audio_path, "rb") as f:
        result = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="verbose_json",  # Returns segments with timestamps
            timestamp_granularities=["segment"]
        )
    return {
        "text": result.text,
        "segments": [
            {"start": s.start, "end": s.end, "text": s.text}
            for s in result.segments
        ]
    }

def translate_audio_to_english(audio_path: str) -> str:
    """Translate non-English audio directly to English text."""
    with open(audio_path, "rb") as f:
        return client.audio.translations.create(
            model="whisper-1",
            file=f,
            response_format="text"
        )

# Supported formats: mp3, mp4, mpeg, mpga, m4a, wav, webm
# Max file size: 25 MB


if __name__ == "__main__":
    print(transcribe("/Users/volod34/PycharmProjects/my-ai-project/week_09/audio.mp3", 'en'))
    print(transcribe_with_timestamps("/Users/volod34/PycharmProjects/my-ai-project/week_09/audio.mp3"))
    print(translate_audio_to_english("/Users/volod34/PycharmProjects/my-ai-project/week_09/audio.mp3"))
