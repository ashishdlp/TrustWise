from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn.functional as F
import sqlite3
from datetime import datetime, timedelta
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi

app = FastAPI(
    docs_url=None,  # Disable Swagger UI
    redoc_url=None,  # Disable ReDoc
    openapi_url=None,  # Disable OpenAPI schema
    title="Text Analysis Dashboard",
    description="A secure text analysis service",
    version="1.0.0"
)

# Serve static files
app.mount("/static", StaticFiles(directory="."), name="static")

# Model configurations
gibberish_model = AutoModelForSequenceClassification.from_pretrained("wajidlinux99/gibberish-text-detector")
gibberish_tokenizer = AutoTokenizer.from_pretrained("wajidlinux99/gibberish-text-detector")
education_model = AutoModelForSequenceClassification.from_pretrained("HuggingFaceFW/fineweb-edu-classifier")
education_tokenizer = AutoTokenizer.from_pretrained("HuggingFaceFW/fineweb-edu-classifier")

class TextInput(BaseModel):
    """
    Pydantic model for text input validation.
    
    Attributes:
        text (str): The input text to be analyzed
    """
    text: str

def ensure_database():
    """
    Creates the SQLite database and required table if they don't exist.
    
    Table Schema:
    - PrimaryKey: Unique identifier for each record
    - Timestamp: When the analysis was performed
    - InputText: The text that was analyzed
    - Gibberish_Clean: Score for clean text classification
    - Gibberish_MildGibberish: Score for mild gibberish classification
    - Gibberish_Noise: Score for noise classification
    - Gibberish_WordSalad: Score for word salad classification
    - Gibberish_FinalCategory: The highest scoring gibberish category
    - Gibberish_FinalScore: The score for the final category
    - Education_Score: Educational content score
    """
    conn = sqlite3.connect('/app/data/text_analysis.db')
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
    """
    Analyzes input text using both gibberish and educational content models.
    
    Args:
        text (str): The input text to analyze
        
    Returns:
        dict: Contains two keys:
            - 'Gibberish': Dict of scores for each gibberish category
            - 'Education': Dict with binary classification score
            
    Example return value:
    {
        'Gibberish': {
            'clean': 0.8,
            'mild gibberish': 0.1,
            'noise': 0.05,
            'word salad': 0.05
        },
        'Education': {
            'Class 1': 0.75
        }
    }
    """
    # Gibberish Analysis
    gibberish_inputs = gibberish_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        gibberish_outputs = gibberish_model(**gibberish_inputs)
    gibberish_probs = F.softmax(gibberish_outputs.logits, dim=-1)
    
    # Calculate gibberish scores for each category
    gibberish_scores = {
        label: gibberish_probs[0][idx].item() 
        for label, idx in gibberish_model.config.label2id.items()
    }
    
    # Educational Content Analysis
    education_inputs = education_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        education_outputs = education_model(**education_inputs)
    
    # Use sigmoid for binary classification
    education_probs = torch.sigmoid(education_outputs.logits)
    education_score = education_probs[0].item()

    return {
        "Gibberish": gibberish_scores,
        "Education": {"Class 1": education_score},
    }

def log_to_database(text: str, scores: dict):
    """
    Logs analysis results to the SQLite database.
    
    Args:
        text (str): The analyzed text
        scores (dict): Dictionary containing both gibberish and education scores
        
    Note:
        The function automatically determines the final gibberish category
        based on the highest score among all categories.
    """
    ensure_database()  # Ensure database exists before logging
    
    conn = sqlite3.connect('/app/data/text_analysis.db')
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
    """
    Endpoint for text analysis.
    
    Args:
        text_input (TextInput): Pydantic model containing the text to analyze
        
    Returns:
        dict: Contains the original text and analysis scores
        
    Raises:
        HTTPException: 500 error if analysis fails
    """
    try:
        scores = score_text(text_input.text)
        log_to_database(text_input.text, scores)
        return {
            "received_text": text_input.text, 
            "scores": {
                "Gibberish": {k.title(): scores["Gibberish"][k] for k in scores["Gibberish"]},
                "Education": scores["Education"]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history")
async def get_scoring_history(start_date: str = None, end_date: str = None):
    """
    Retrieves analysis history within the specified date range.
    
    Args:
        start_date (str, optional): Start date in ISO format (YYYY-MM-DD)
        end_date (str, optional): End date in ISO format (YYYY-MM-DD)
        
    Returns:
        list: List of dictionaries containing analysis records
        
    Raises:
        HTTPException: 500 error if database query fails
    """
    try:
        conn = sqlite3.connect('/app/data/text_analysis.db')
        cursor = conn.cursor()
        
        query = "SELECT * FROM text_scores"
        params = []
        
        if start_date and end_date:
            query += " WHERE Timestamp BETWEEN ? AND ?"
            params.extend([start_date, end_date])
            
        query += " ORDER BY Timestamp DESC LIMIT 100"
        
        cursor.execute(query, params)
        columns = [column[0] for column in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        conn.close()
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Add a catch-all route for undefined endpoints
@app.api_route("/{path_name:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def catch_all(path_name: str):
    raise HTTPException(status_code=404, detail="Resource not found")
