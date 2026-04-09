import json
from datetime import datetime
from chatbot_class import Chatbot

class PersistentChatbot(Chatbot):
    def save_conversation(self, filename: str = None):
        """Save conversation to a JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"conversation_{timestamp}.json"

        data = {
            "system_prompt": self.system_prompt,
            "messages": self.messages,
            "saved_at": datetime.now().isoformat()
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Saved to {filename}")
        return filename

    @classmethod
    def load_conversation(cls, filename: str):
        """Load a conversation from a JSON file."""
        with open(filename, 'r') as f:
            data = json.load(f)

        bot = cls(data["system_prompt"])
        bot.messages = data["messages"]

        print(f"Loaded conversation with {len(bot.messages)-1} messages")
        return bot

# Test it
if __name__ == "__main__":
    bot = PersistentChatbot("You are a storyteller.")

    bot.chat("Start a story about a dragon")
    bot.chat("What happens next?")

    filename = bot.save_conversation()

    bot2 = PersistentChatbot.load_conversation(filename)
    print(bot2.chat("Continue the story from where you left off"))
