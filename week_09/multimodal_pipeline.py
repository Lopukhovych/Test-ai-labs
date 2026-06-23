# multimodal_pipeline.py
"""
A complete multi-modal pipeline:
1. Receive voice question (audio)
2. Transcribe with Whisper
3. Optionally analyze an image
4. Generate AI response
5. Speak the response (TTS)
"""
from openai import OpenAI
from pathlib import Path
import base64
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

class MultiModalAssistant:
    def __init__(self, system_prompt: str = "You are a helpful assistant."):
        self.system_prompt = system_prompt
        self.history = []

    def transcribe(self, audio_path: str) -> str:
        """Voice → text."""
        with open(audio_path, "rb") as f:
            return client.audio.transcriptions.create(
                model="whisper-1", file=f, response_format="text"
            )

    def analyze_image(self, image_path: str) -> str:
        """Image → text description."""
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image concisely."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                ]
            }],
            max_tokens=500
        )
        return response.choices[0].message.content

    def chat(self, text_input: str, image_context: str = None) -> str:
        """Generate text response with optional image context."""
        user_content = text_input
        if image_context:
            user_content = f"[Image context: {image_context}]\n\nUser question: {text_input}"

        self.history.append({"role": "user", "content": user_content})

        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": self.system_prompt},
                *self.history
            ]
        )
        reply = response.choices[0].message.content
        self.history.append({"role": "assistant", "content": reply})
        return reply

    def speak(self, text: str, output_path: str) -> str:
        """Text → audio file."""
        response = client.audio.speech.create(
            model="tts-1", voice="nova", input=text
        )
        with open(output_path, "wb") as f:
            f.write(response.content)
        return output_path

    def voice_chat(self, audio_path: str, image_path: str = None) -> dict:
        """Full pipeline: audio in → audio out."""
        # 1. Transcribe question
        question = self.transcribe(audio_path)
        print(f"You said: {question}")

        # 2. Optionally analyze image
        image_context = None
        if image_path:
            image_context = self.analyze_image(image_path)
            print(f"Image context: {image_context}")

        # 3. Generate response
        answer = self.chat(question, image_context)
        print(f"Assistant: {answer}")

        # 4. Speak response
        output_audio = "response.mp3"
        self.speak(answer, output_audio)

        return {
            "question": question,
            "answer": answer,
            "audio": output_audio
        }

# Usage:
assistant = MultiModalAssistant("You are a helpful assistant.")
output_path = assistant.speak("I am achromate. Analyze my dog and describe me her appearance with colors", output_path="request.mp3")
print(f'output_path: {output_path}')
# request_test = assistant.transcribe(output_path)
# print(f'request_test: {request_test}')
result = assistant.voice_chat(output_path, image_path="/Users/volod34/PycharmProjects/my-ai-project/week_09/polly.jpg")
print(f'result: {result}')
