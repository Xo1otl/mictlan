import google.generativeai as genai
from infra.ai import llm

genai.configure(api_key=llm.GOOGLE_CLOUD_API_KEY)
model = genai.GenerativeModel('gemini-exp-1121')
chat = model.start_chat(
    history=[
        {"role": "user", "parts": "Hello"},
        {"role": "model", "parts": "Great to meet you. What would you like to know?"},
    ] # type: ignore
)
response = chat.send_message("I have 2 dogs in my house.", stream=True)
for chunk in response:
    print(chunk.text)
    print("_" * 80)
response = chat.send_message("How many paws are in my house?", stream=True)
for chunk in response:
    print(chunk.text)
    print("_" * 80)

print(chat.history)
