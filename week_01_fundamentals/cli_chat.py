from chatbot_class import Chatbot
import sys

def print_help():
    print("""
Commands:
  /clear     - Clear conversation history
  /history   - Show conversation history
  /system    - Change system prompt
  /quit      - Exit the chatbot

Just type normally to chat!
""")

def main():
    print("=" * 50)
    print("  Welcome to AI Chatbot!")
    print("=" * 50)
    print("Type /help for commands\n")

    bot = Chatbot("You are a helpful and friendly assistant.")

    while True:
        try:
            user_input = input("You: ").strip()
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        # Handle commands
        if user_input.startswith("/"):
            command = user_input.lower()

            if command == "/quit":
                print("Goodbye!")
                break
            elif command == "/clear":
                bot.clear_history()
                print("Conversation cleared.\n")
            elif command == "/history":
                history = bot.get_history()
                if not history:
                    print("No conversation yet.\n")
                else:
                    for msg in history:
                        role = "You" if msg["role"] == "user" else "AI"
                        print(f"{role}: {msg['content'][:100]}...")
                    print()
            elif command == "/help":
                print_help()
            elif command.startswith("/system "):
                new_prompt = user_input[8:]
                bot = Chatbot(new_prompt)
                print(f"System prompt changed.\n")
            else:
                print(f"Unknown command: {command}")
                print_help()
        else:
            # Normal chat
            response = bot.chat(user_input)
            print(f"AI: {response}\n")

if __name__ == "__main__":
    main()
