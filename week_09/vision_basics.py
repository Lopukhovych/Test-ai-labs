# vision_basics.py
from openai import OpenAI
import base64
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()
# llm = OpenAI(model="gpt-5-mini")

def encode_image(path: str) -> str:
    """Encode image to base64 for the API."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def analyze_image(image_path: str, question: str = "What is in this image?") -> str:
    """Send an image to gpt-5-mini for analysis."""
    b64 = encode_image(image_path)
    print(f'b64: {b64}')

    # Detect format from extension
    ext = Path(image_path).suffix.lower().lstrip(".")
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
            "gif": "image/gif", "webp": "image/webp"}.get(ext, "image/jpeg")
    print(f'mime: {mime}')

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
            ]
        }],
        max_tokens=500
    )
    return response.choices[0].message.content

def analyze_image_url(url: str, question: str) -> str:
    """Analyze an image from a URL (no base64 needed)."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": url}}
            ]
        }],
        max_tokens=500
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    # Local file
    print(analyze_image("/Users/volod34/PycharmProjects/my-ai-project/week_09/orchestrator_worker.png", "Describe what you see"))

    # URL
    # url = "https://upload.wikimedia.org/wikipedia/commons/thumb/b/ba/Flower_jtca001.jpg/960px-Flower_jtca001.jpg"
    # print(analyze_image_url(url, "What flower is this? What color is it?"))
