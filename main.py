from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn.functional as F
import sqlite3
from datetime import datetime

app = FastAPI()

# Serve static files
app.mount("/static", StaticFiles(directory="."), name="static")

# Load models and tokenizers
gibberish_model = AutoModelForSequenceClassification.from_pretrained("wajidlinux99/gibberish-text-detector")
gibberish_tokenizer = AutoTokenizer.from_pretrained("wajidlinux99/gibberish-text-detector")
education_model = AutoModelForSequenceClassification.from_pretrained("HuggingFaceFW/fineweb-edu-classifier")
education_tokenizer = AutoTokenizer.from_pretrained("HuggingFaceFW/fineweb-edu-classifier")

class TextInput(BaseModel):
    text: str

def score_text(text: str):
    # Gibberish Score
    gibberish_inputs = gibberish_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        gibberish_outputs = gibberish_model(**gibberish_inputs)
    gibberish_probs = F.softmax(gibberish_outputs.logits, dim=-1)
    gibberish_scores = {label: gibberish_probs[0][idx].item() for label, idx in gibberish_model.config.label2id.items()}

    # Education Score
    education_inputs = education_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        education_outputs = education_model(**education_inputs)
    education_probs = F.softmax(education_outputs.logits, dim=-1)
    education_scores = {f"Class {idx}": score.item() for idx, score in enumerate(education_probs[0])}

    return {
        "Gibberish": gibberish_scores,
        "Education": education_scores,
    }

def log_to_database(text: str, scores: dict):
    conn = sqlite3.connect('text_analysis.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS text_scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME,
            input_text TEXT,
            gibberish_clean REAL,
            gibberish_gibberish REAL,
            education_class_0 REAL,
            education_class_1 REAL
        )
    ''')
    
    now = datetime.now()
    cursor.execute('''
        INSERT INTO text_scores (
            timestamp, input_text, 
            gibberish_clean, gibberish_gibberish,
            education_class_0, education_class_1
        ) VALUES (?, ?, ?, ?, ?, ?)
    ''', (
        now, 
        text,
        scores['Gibberish'].get('clean', 0),
        scores['Gibberish'].get('gibberish', 0),
        scores['Education'].get('Class 0', 0),
        scores['Education'].get('Class 1', 0)
    ))
    
    conn.commit()
    conn.close()

@app.get("/")
async def read_index():
    return FileResponse("index.html")

@app.post("/score_text/")
async def score_text_endpoint(text_input: TextInput):
    try:
        scores = score_text(text_input.text)
        log_to_database(text_input.text, scores)
        return {"received_text": text_input.text, "scores": scores}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history")
async def get_scoring_history():
    conn = sqlite3.connect('text_analysis.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM text_scores ORDER BY timestamp DESC LIMIT 100")
    columns = [column[0] for column in cursor.description]
    results = [dict(zip(columns, row)) for row in cursor.fetchall()]
    conn.close()
    return results