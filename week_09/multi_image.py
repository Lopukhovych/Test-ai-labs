# multi_image.py
from openai import OpenAI
import base64
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

def compare_images(image_paths: list[str], question: str) -> str:
    """Send multiple images in one request for comparison."""
    content = [{"type": "text", "text": question}]

    for path in image_paths:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"}
        })

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": content}],
        max_tokens=500
    )
    return response.choices[0].message.content

# Usage: compare_images(["v1.png", "v2.png"], "What changed between these UI designs?")


if __name__ == "__main__":
    answer = compare_images([
        "/Users/volod34/PycharmProjects/my-ai-project/week_09/evaluator_optimizer.png",
        "/Users/volod34/PycharmProjects/my-ai-project/week_09/human_in_loop_graph.png"
    ], "What changed between these two files?")
    print(f'answer: {answer}')
