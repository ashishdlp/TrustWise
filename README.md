# Text Analysis Dashboard

A basic python based web application that does real-time analysis of text content, detecting gibberish and evaluating educational content using machine learning models.

## Features

- **Gibberish Detection**: Analyzes text and classifies it into four categories:
  - Clean text
  - Mild gibberish
  - Noise
  - Word salad

- **Educational Content Scoring**: Evaluates the educational value of the provided text

- **Historical Analysis**: Maintains a database of all analyses with timestamp tracking

- **Web Interface**: Clean and intuitive dashboard for text analysis

- **RESTful API**: Secure endpoints for programmatic access

## Technologies Used

- **Backend**: FastAPI (Python 3.10)
- **Machine Learning**: PyTorch, Hugging Face Transformers
- **Database**: SQLite
- **Frontend**: HTML/CSS/JavaScript
- **Containerization**: Docker

## Installation

### Local Setup

1. Clone the repository:
```bash
git clone https://github.com/ashishdlp/TrustWise
cd TrustWise
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
uvicorn main:app --reload
```

The application will be available at `http://localhost:8000`

### Docker Setup

1. Build the Docker image:
```bash
docker build -t text-analysis-dashboard .
```

2. Run the container:
```bash
docker run -p 8000:8000 text-analysis-dashboard
```

## Usage

1. Open your web browser and navigate to `http://localhost:8000`
2. Enter the text you want to analyze in the input field
3. Submit the text to receive analysis results including:
   - Gibberish detection scores
   - Educational content evaluation
4. View historical analyses in the dashboard

## API Endpoints

### POST /score_text/
Analyzes provided text and returns scores.

**Request Body:**
```json
{
    "text": "Your text here"
}
```

**Response:**
```json
{
    "received_text": "Your text here",
    "scores": {
        "Gibberish": {
            "Clean": 0.8,
            "Mild Gibberish": 0.1,
            "Noise": 0.05,
            "Word Salad": 0.05
        },
        "Education": {
            "Class 1": 0.75
        }
    }
}
```

### GET /history
Retrieves analysis history.

**Query Parameters:**
- `start_date` (optional): Start date in ISO format (YYYY-MM-DD)
- `end_date` (optional): End date in ISO format (YYYY-MM-DD)

## Database Schema

The SQLite database (`text_analysis.db`) contains the following schema:

```sql
CREATE TABLE text_scores (
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
```

## Models

The application uses two pre-trained models from Hugging Face:
- `wajidlinux99/gibberish-text-detector`: For gibberish detection
- `HuggingFaceFW/fineweb-edu-classifier`: For educational content classification

## Security

- API documentation (Swagger/ReDoc) is disabled by default
- Implements proper error handling and input validation
- Uses Pydantic models for request validation
