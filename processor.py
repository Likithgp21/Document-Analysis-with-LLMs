import PyPDF2
import nltk
from engine import summarize_text, classify_text, extract_entities

def load_pdf_text(pdf_path):
    """Extracts text from a PDF file."""
    try:
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None

def chunk_text(text, max_sentences=10):
    """Splits text into chunks of `max_sentences` each."""
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = []
    
    for sentence in sentences:
        current_chunk.append(sentence)
        if len(current_chunk) >= max_sentences:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            
    if current_chunk:  # Add the last chunk if any
        chunks.append(" ".join(current_chunk))
        
    return chunks

def process_large_document(file_path):
    """
    Main processing pipeline for a large document.
    Implements a "Map-Reduce" strategy for summarization.
    """
    print(f"Processing document: {file_path}")
    
    # 1. Load and Chunk
    document_text = load_pdf_text(file_path)
    if not document_text:
        return {"error": "Could not read text from PDF."}
    
    text_chunks = chunk_text(document_text)
    
    # 2. Map-Reduce Summarization
    print(f"Summarizing {len(text_chunks)} chunks...")
    chunk_summaries = []
    for i, chunk in enumerate(text_chunks):
        print(f"Summarizing chunk {i+1}/{len(text_chunks)}")
        # MAP: Summarize each individual chunk
        chunk_summary = summarize_text(chunk)
        chunk_summaries.append(chunk_summary)
        
    # REDUCE: Combine all chunk summaries and summarize *that*
    combined_summary_text = " ".join(chunk_summaries)
    
    print("Creating final summary...")
    final_summary = summarize_text(combined_summary_text)

    # 3. Analyze the final summary (more efficient than analyzing the whole doc)
    print("Extracting entities and categories...")
    
    # Extract Entities from the summary
    entities = extract_entities(final_summary)
    
    # Categorize the summary
    candidate_labels = ["Technology", "Business", "Legal", "Finance", "Academic Paper", "Marketing"]
    categories = classify_text(final_summary, candidate_labels)
    
    return {
        "final_summary": final_summary,
        "categories": categories,
        "entities": entities,
        "original_chunk_count": len(text_chunks)
    }

if __name__ == '__main__':
    # Create a dummy PDF for testing if you don't have one
    # For now, this will fail unless you have 'test.pdf'
    # print(process_large_document('test.pdf'))
    print("Processor module loaded. Ready to be used by Flask.")