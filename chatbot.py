import os
from dotenv import load_dotenv
from google.generativeai import genai

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel(model_name="models/gemini-2.0-flash")
response = model.generate_content("Say something calming.")
print(response.text)
