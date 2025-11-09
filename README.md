Document Analysis with LLMs

This project is a Python-based web application that uses Hugging Face transformers (PyTorch) and Flask to analyze large PDF documents. Upload a PDF and receive a JSON object containing a concise summary, extracted named entities (People, Organizations, etc.), and a list of relevant categories.

**Features**

PDF Upload: Simple web interface to upload .pdf files.

AI Summarization: Generates a short, readable summary of the entire document.

Zero-Shot Categorization: Classifies the document into a list of predefined categories (e.g., "Technology," "Finance," "Legal") without needing to be trained on them.

Named Entity Recognition (NER): Extracts key entities like people, organizations, locations, and dates.

Large Document Handling: Implements a "Map-Reduce" (chunking) strategy to analyze documents far larger than the token limits of the underlying models.

How It Works: Architecture

The system is broken into three main components:

Frontend (index.html): A basic HTML/JS page that provides the file upload form and displays the final JSON result.

Backend API (app.py): A Flask server that handles file uploads, saves the temporary file, and orchestrates the analysis process.

ML Engine:

processor.py: This file contains the business logic. It uses PyPDF2 to read the PDF and NLTK to chunk the text. It then implements the Map-Reduce logic, first summarizing all chunks, then creating a final summary of those summaries.

engine.py: This file loads the three pre-trained PyTorch models from Hugging Face (summarizer, categorizer, extractor) and wraps them in simple Python functions.

Project Structure

DocAnalysis-LLM/
├── app.py              # The Flask server

├── engine.py           # Loads the PyTorch models (Summarizer, NER, etc.)

├── processor.py        # Handles PDF reading, chunking, and Map-Reduce logic

├── requirements.txt    # Python dependencies

├── templates/
│   └── index.html      # The frontend webpage

└── uploads/             # Temporary folder for uploaded files (auto-created)


Setup and Installation

1. Clone the repository:

git clone [https://github.com/your-username/DocAnalysis-LLM.git](https://github.com/your-username/DocAnalysis-LLM.git)
cd DocAnalysis-LLM


2. Create and activate a virtual environment:

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate


3. Install dependencies:

First, install the packages from requirements.txt.

pip install -r requirements.txt


4. Download NLTK data:

The text chunker (processor.py) requires the punkt tokenizer from NLTK.

python -c "import nltk; nltk.download('punkt')"


requirements.txt

flask
torch
transformers
sentencepiece
PyPDF2
nltk


How to Run

Activate your virtual environment:

source venv/bin/activate


Run the Flask application:

python app.py


The server will start, and the PyTorch models will be downloaded and loaded into memory (this may take a moment on first run).

Open your browser:
Navigate to http://127.0.0.1:5000.

Upload a PDF:
Use the form to select a PDF file and click "Analyze Document." Wait for the processing to complete, and the analysis will appear on the page.
