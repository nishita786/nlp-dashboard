from fastapi import FastAPI
from pydantic import BaseModel
import spacy
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
from collections import Counter

# Initialize app
app = FastAPI()

# ✅ CORS (frontend connection fix)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
nlp = spacy.load("en_core_web_sm")
sentiment_model = pipeline("sentiment-analysis")

# Request schema
class TextInput(BaseModel):
    text: str

# Root
@app.get("/")
def home():
    return {"message": "Backend running 🚀"}

# Main API
@app.post("/analyze")
def analyze_text(input: TextInput):
    doc = nlp(input.text)

    # 🔹 Entities
    entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]

    # 🔹 Sentiment (Transformer)
    result_sentiment = sentiment_model(input.text)[0]
    sentiment = {
        "label": result_sentiment["label"],
        "score": round(result_sentiment["score"], 3)
    }

    # 🔹 Keywords with frequency
    words = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN"]]
    freq = Counter(words)
    keywords = [{"word": k, "count": v} for k, v in freq.items()]

    return {
        "entities": entities,
        "sentiment": sentiment,
        "keywords": keywords
    }