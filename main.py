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

def ensure_database():
    """Ensure the database and table exist with the correct schema."""
    conn = sqlite3.connect('text_analysis.db')
    cursor = conn.cursor()
    
    # Check if table exists, if not create it
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS text_scores (
            PrimaryKey INTEGER PRIMARY KEY AUTOINCREMENT,
            Timestamp DATETIME,
            InputText TEXT,
            Gibberish_Clean REAL,
            Gibberish_MildGibberish REAL,
            Gibberish_Noise REAL,
            Gibberish_WordSalad REAL,
            Gibberish_FinalCategory TEXT,
            Gibberish_FinalScore REAL,
            Education_Score REAL
        )
    ''')
    conn.commit()
    conn.close()

def score_text(text: str):
    # Gibberish Score
    gibberish_inputs = gibberish_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        gibberish_outputs = gibberish_model(**gibberish_inputs)
    gibberish_probs = F.softmax(gibberish_outputs.logits, dim=-1)
    
    # Get all gibberish scores
    gibberish_scores = {
        label: gibberish_probs[0][idx].item() 
        for label, idx in gibberish_model.config.label2id.items()
    }
    
    # Find the category with the highest score
    highest_gibberish_category = max(gibberish_scores, key=gibberish_scores.get)
    gibberish_result = {highest_gibberish_category: gibberish_scores[highest_gibberish_category]}

    # Education Score
    education_inputs = education_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        education_outputs = education_model(**education_inputs)
    education_probs = F.softmax(education_outputs.logits, dim=-1)
    education_scores = {f"Class {idx}": score.item() for idx, score in enumerate(education_probs[0])}

    return {
        "Gibberish": gibberish_scores,  # Return full scores
        "Education": education_scores,
    }

def log_to_database(text: str, scores: dict):
    ensure_database()  # Ensure database exists before logging
    
    conn = sqlite3.connect('text_analysis.db')
    cursor = conn.cursor()
    
    now = datetime.now()
    
    # Prepare gibberish scores
    final_category = max(scores['Gibberish'], key=scores['Gibberish'].get)
    final_score = scores['Gibberish'][final_category]
    
    cursor.execute('''
        INSERT INTO text_scores (
            Timestamp, InputText, 
            Gibberish_Clean, Gibberish_MildGibberish, 
            Gibberish_Noise, Gibberish_WordSalad,
            Gibberish_FinalCategory, Gibberish_FinalScore,
            Education_Score
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        now, 
        text,
        scores['Gibberish'].get('clean', 0),
        scores['Gibberish'].get('mild gibberish', 0),
        scores['Gibberish'].get('noise', 0),
        scores['Gibberish'].get('word salad', 0),
        final_category,
        final_score,
        scores['Education'].get('Class 1', 0)
    ))
    
    conn.commit()
    conn.close()

@app.on_event("startup")
async def startup_event():
    ensure_database()

@app.get("/")
async def read_index():
    return FileResponse("index.html")

@app.post("/score_text/")
async def score_text_endpoint(text_input: TextInput):
    try:
        scores = score_text(text_input.text)
        log_to_database(text_input.text, scores)
        return {
            "received_text": text_input.text, 
            "scores": {
                "Gibberish": {k: scores["Gibberish"][k] for k in scores["Gibberish"]},
                "Education": scores["Education"]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history")
async def get_scoring_history():
    try:
        conn = sqlite3.connect('text_analysis.db')
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM text_scores ORDER BY Timestamp DESC LIMIT 100")
        columns = [column[0] for column in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        conn.close()
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))