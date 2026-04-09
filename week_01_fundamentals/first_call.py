# from openai import OpenAI
# from dotenv import load_dotenv
#
# load_dotenv()
# client = OpenAI()
#
# print("Chat with AI! Type 'quit' to exit.\n")
#
# while True:
#     # Get user input
#     user_input = input("You: ")
#
#     # Check if user wants to quit
#     if user_input.lower() == 'quit':
#         print("Goodbye!")
#         break
#
#     # Send to AI
#     response = client.chat.completions.create(
#         model="gpt-5-mini",
#         messages=[{"role": "user", "content": user_input}]
#     )
#
#     # Print AI response
#     print(f"AI: {response.choices[0].message.content}\n")
from chatbot_class import Chatbot

if __name__ == "__main__":
    bot = Chatbot("You are a friendly assistant who loves jokes.")

    print(bot.chat("Tell me a joke"))
