# TrustWise Project

TrustWise is a Python-based application with web interface capabilities, containerized using Docker.

## Features
- Python backend functionality
- Web interface (index.html)
- Docker containerization
- Database integration (text_analysis.db)

## Installation

### Prerequisites
- Python 3.11+
- Docker (optional)

### Local Installation
1. Clone this repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Docker Installation
1. Build the Docker image:
   ```bash
   docker build -t trustwise .
   ```
2. Run the container:
   ```bash
   docker run -p 5000:5000 trustwise
   ```

## Usage
Run the application:
```bash
python main.py
```

Access the web interface at `http://localhost:5000`

## Project Structure
```
.
├── .dockerignore
├── Dockerfile
├── README.md
├── index.html
├── main.py
├── postcss.config.js
├── requirements.txt
├── text_analysis.db
```

## Requirements
See [requirements.txt](requirements.txt) for the complete list of Python dependencies.

## License
[MIT License](LICENSE)
