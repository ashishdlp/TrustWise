from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.nn.functional as F
import sqlite3
from datetime import datetime
from fastapi.responses import FileResponse
app = FastAPI()


# Serve static files (CSS, JS) from the current directory
app.mount("/static", StaticFiles(directory="."), name="static")

# Load models and tokenizers
gibberish_model = AutoModelForSequenceClassification.from_pretrained("wajidlinux99/gibberish-text-detector")
gibberish_tokenizer = AutoTokenizer.from_pretrained("wajidlinux99/gibberish-text-detector")
education_model = AutoModelForSequenceClassification.from_pretrained("HuggingFaceFW/fineweb-edu-classifier")
education_tokenizer = AutoTokenizer.from_pretrained("HuggingFaceFW/fineweb-edu-classifier")


class TextInput(BaseModel):
    text: str

def get_scores(text: str):
    # Gibberish Score
    gibberish_inputs = gibberish_tokenizer(text, return_tensors="pt")
    gibberish_outputs = gibberish_model(**gibberish_inputs)
    gibberish_probs = F.softmax(gibberish_outputs.logits, dim=-1)
    gibberish_scores = {label: gibberish_probs[0][idx].item() for label, idx in gibberish_model.config.label2id.items()}

    # Education Score
    education_inputs = education_tokenizer(text, return_tensors="pt")
    education_outputs = education_model(**education_inputs)
    education_probs = F.softmax(education_outputs.logits, dim=-1)
    education_scores = {f"Class {idx}": score.item() for idx, score in enumerate(education_probs[0])}


    return {
        "Gibberish": gibberish_scores,
        "Education": education_scores,
    }


def log_to_db(text: str, scores: dict):
    conn = sqlite3.connect('text_scores.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME,
            text TEXT,
            gibberish_clean REAL,
            gibberish_gibberish REAL,
            education_class_0 REAL,
            education_class_1 REAL,
            toxicity_toxic REAL,
            toxicity_nontoxic REAL,
            emotion_anger REAL,
            emotion_disgust REAL,
            emotion_fear REAL,
            emotion_joy REAL,
            emotion_neutral REAL,
            emotion_sadness REAL,
            emotion_surprise REAL,
            vectara REAL
        )
    ''')
    now = datetime.now()
    cursor.execute('''
        INSERT INTO scores (timestamp, text, gibberish_clean, gibberish_gibberish, education_class_0, education_class_1, toxicity_toxic, toxicity_nontoxic, emotion_anger, emotion_disgust, emotion_fear, emotion_joy, emotion_neutral, emotion_sadness, emotion_surprise, vectara)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        now,
        text,
        scores["Gibberish"].get("clean", 0),
        scores["Gibberish"].get("gibberish", 0),
        scores["Education"].get("Class 0", 0), # Assuming Class 0 and Class 1, adjust if labels are different
        scores["Education"].get("Class 1", 0),
        0, # toxicity_toxic
        0, # toxicity_nontoxic
        0, # emotion_anger
        0, # emotion_disgust
        0, # emotion_fear
        0, # emotion_joy
        0, # emotion_neutral
        0, # emotion_sadness
        0, # emotion_surprise
        0 # vectara
    ))
    conn.commit()
    conn.close()

@app.get("/")
async def read_index():
    return FileResponse("index.html")

@app.post("/score_text/")
async def score_text_endpoint(text_input: TextInput):
    try:
        scores = get_scores(text_input.text)
        log_to_db(text_input.text, scores)
        return {"received_text": text_input.text, "scores": scores}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
