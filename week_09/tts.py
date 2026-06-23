# tts.py
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

# Available voices: alloy, ash, coral, echo, fable, onyx, nova, sage, shimmer
VOICES = {
    "neutral": "alloy",
    "professional": "onyx",
    "friendly": "nova",
    "warm": "shimmer",
}

def speak(text: str, output_path: str, voice: str = "alloy", speed: float = 1.0) -> str:
    """Convert text to speech and save to file."""
    response = client.audio.speech.create(
        model="tts-1",        # tts-1-hd for higher quality
        voice=voice,
        input=text,
        speed=speed           # 0.25 to 4.0
    )
    with open(output_path, "wb") as f:
        f.write(response.content)
    return output_path

def speak_ssml(text: str, output_path: str) -> str:
    """Use tts-1-hd for higher quality output."""
    response = client.audio.speech.create(
        model="tts-1-hd",
        voice="nova",
        input=text
    )
    with open(output_path, "wb") as f:
        f.write(response.content)
    return output_path

if __name__ == "__main__":
    for i, voice in enumerate(VOICES.values()):
        speak("Hello! Welcome to the AI Engineering course.", f"greeting_{voice}.mp3", voice=voice)
    print("Audio saved to greeting.mp3")
