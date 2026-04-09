from chatbot_class import Chatbot

class SmartChatbot(Chatbot):
    def __init__(self, system_prompt: str, max_messages: int = 20):
        super().__init__(system_prompt)
        self.max_messages = max_messages

    def chat(self, user_message: str) -> str:
        # Check if we need to summarize
        if len(self.messages) > self.max_messages:
            self._summarize_history()

        # Normal chat
        return super().chat(user_message)

    def _summarize_history(self):
        """Compress conversation history into a summary."""
        history = self.messages[1:]

        history_text = "\n".join([
            f"{m['role']}: {m['content']}"
            for m in history
        ])

        summary_response = self.client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": "Summarize this conversation in 2-3 sentences."},
                {"role": "user", "content": history_text}
            ]
        )

        summary = summary_response.choices[0].message.content

        self.messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "system", "content": f"Previous conversation summary: {summary}"}
        ]

        print(f"[Summarized {len(history)} messages]")

# Test
if __name__ == "__main__":
    bot = SmartChatbot("You are a helpful assistant.", max_messages=6)

    for i in range(10):
        response = bot.chat(f"Message number {i+1}")
        print(f"Bot: {response[:50]}...")
