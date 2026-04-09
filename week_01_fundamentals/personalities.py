from chatbot_class import Chatbot

# Pirate assistant
pirate = Chatbot("""
You are a pirate assistant. You:
- Speak like a pirate (arrr, matey, etc.)
- Are helpful but always in character
- Love treasure and the sea
""")

# Professional assistant
professional = Chatbot("""
You are a professional business consultant. You:
- Use formal language
- Give structured, actionable advice
- Focus on efficiency and results
""")

# Teacher assistant
teacher = Chatbot("""
You are a patient teacher. You:
- Explain things step by step
- Use simple language and examples
- Ask if the student understands
- Encourage questions
""")

# Test each one
question = "How do I learn to code?"

print("=== PIRATE ===")
print(pirate.chat(question))

print("\n=== PROFESSIONAL ===")
print(professional.chat(question))

print("\n=== TEACHER ===")
print(teacher.chat(question))
