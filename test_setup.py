import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
model = os.getenv("OPENAI_MODEL")

with open("data/tech_company_corpus.txt", "r", encoding="utf-8") as f:
    corpus = f.read()

print("API key loaded:", bool(api_key))
print("Model:", model)
print("Corpus length:", len(corpus))
print("First 200 chars:")
print(corpus[:200])