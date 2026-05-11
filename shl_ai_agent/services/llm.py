import os

import google.generativeai as genai
from dotenv import load_dotenv


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

model = None

if api_key and api_key != "your_key_here":
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")


def generate_reply(prompt, fallback=None):
    if model is None:
        return fallback or "I found relevant SHL assessments for this hiring need."

    response = model.generate_content(prompt)
    return response.text
