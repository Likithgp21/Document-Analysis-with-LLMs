from flask import Flask, request, jsonify, render_template
import os
from processor import process_large_document

# Initialize Flask app
app = Flask(__name__)

# Configure a temporary upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    """Render the main upload page."""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_document():
    """
    Handle the file upload and analysis.
    """
    if 'document' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['document']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and (file.filename.endswith('.pdf') or file.filename.endswith('.txt')):
        # Save the file temporarily
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        # Process the document
        try:
            analysis_results = process_large_document(file_path)
            # Clean up the temp file
            os.remove(file_path)
            
            return jsonify(analysis_results)
        
        except Exception as e:
            # Clean up the temp file even if an error occurs
            if os.path.exists(file_path):
                os.remove(file_path)
            return jsonify({"error": str(e)}), 500
    
    return jsonify({"error": "Invalid file type. Please upload .pdf or .txt"}), 400

if __name__ == '__main__':
    # Run the app in debug mode (for development)
    app.run(debug=True, port=5000)