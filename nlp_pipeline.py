"""Lightweight NLP pipeline with optional spaCy NER and safe fallback."""
import re
from .feature_engineering import sentences, extract_features
try:
    import spacy
    _nlp = spacy.blank("en")
    _nlp.add_pipe("sentencizer")
except Exception:
    _nlp = None

def process_text(question,response):
    if _nlp:
        doc=_nlp(response)
        claims=[s.text.strip() for s in doc.sents if s.text.strip()]
        entities=[e.text for e in doc.ents]
    else:
        claims=sentences(response); entities=re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b",response)
    return {"claims":claims,"entities":entities,"features":extract_features(question,response)}

def claim_signals(question, claim, predictor):
    return predictor.predict_one(question, claim)
