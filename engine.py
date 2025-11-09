from transformers import pipeline
import torch

# Use a specific device (GPU if available, else CPU)
device = 0 if torch.cuda.is_available() else -1

# 1. Summarization Pipeline
# We use a model fine-tuned for summarization
summarizer = pipeline(
    "summarization", 
    model="sshleifer/distilbart-cnn-12-6", 
    device=device
)

# 2. Categorization Pipeline
# We use a "zero-shot" model that can classify text into labels you provide
categorizer = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=device
)

# 3. Entity Extraction Pipeline (NER)
# We use a Named Entity Recognition (NER) model
extractor = pipeline(
    "ner", 
    model="dslim/bert-base-NER", 
    grouped_entities=True, # Groups related words (e.g., "New" and "York")
    device=device
)

def summarize_text(text):
    """
    Summarizes a piece of text.
    Handles potential errors for very short text.
    """
    try:
        # The model has a max length, but for chunks, we set a reasonable min/max
        result = summarizer(text, max_length=150, min_length=30, do_sample=False)
        return result[0]['summary_text']
    except Exception as e:
        print(f"Error in summarization: {e}")
        return text  # Return original text if summarization fails

def classify_text(text, candidate_labels):
    """
    Classifies a piece of text using zero-shot learning.
    """
    if not text or not candidate_labels:
        return {}
    
    result = categorizer(text, candidate_labels)
    # Return a dictionary of labels and their scores
    return dict(zip(result['labels'], result['scores']))

def extract_entities(text):
    """
    Extracts named entities from a piece of text.
    """
    try:
        entities = extractor(text)
        # Clean up the output to be more readable
        return [{e['entity_group']: e['word']} for e in entities]
    except Exception as e:
        print(f"Error in entity extraction: {e}")
        return []

if __name__ == '__main__':
    # Test the functions
    sample_text = """
    Sundar Pichai, CEO of Google, announced the new Pixel 8 smartphone 
    at an event in New York City on October 4, 2023. 
    The company, Alphabet Inc., reported strong earnings for the third quarter.
    """
    
    print("--- SUMMARY ---")
    print(summarize_text(sample_text))
    
    print("\n--- CATEGORIES ---")
    labels = ["Technology", "Finance", "Politics"]
    print(classify_text(sample_text, labels))
    
    print("\n--- ENTITIES ---")
    print(extract_entities(sample_text))