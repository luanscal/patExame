import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load models once
nlp = spacy.load("en_core_sci_sm", disable=["tagger", "parser", "ner"])
embedder = SentenceTransformer("all-mpnet-base-v2")

def extract_features(text):
    """Extract technical terms using spaCy"""
    doc = nlp(text)
    return [chunk.text for chunk in doc.noun_chunks]

def get_embeddings(text):
    """Convert text to embeddings"""
    return embedder.encode(text)

def check_compliance(text):
    """Rule-based compliance checks"""
    issues = []
    if "algorithm" in text.lower() and "hardware" not in text.lower():
        issues.append("Potential abstract idea under 35 U.S.C. § 101")
    if "wherein" not in text.lower():
        issues.append("Claims may lack specificity (missing 'wherein')")
    return issues