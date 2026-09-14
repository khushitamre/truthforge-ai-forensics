"""Deterministic linguistic and semantic features for response forensics."""
import re
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

UNCERTAINTY = {"maybe","might","possibly","perhaps","could","likely","approximately"," reportedly","unknown"}
CERTAINTY = {"definitely","always","never","certainly","proven","undeniable","guaranteed"}

def sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text.strip()) if s.strip()]

def tokens(text):
    return re.findall(r"[A-Za-z0-9']+", text.lower())

def _overlap(a,b):
    sa, sb = set(tokens(a)), set(tokens(b))
    return len(sa & sb) / max(1, len(sa | sb))

def extract_features(question, response):
    ss = sentences(response)
    rt = tokens(response); qt = tokens(question)
    nums = re.findall(r"\b\d+(?:\.\d+)?%?\b", response)
    dates = re.findall(r"\b(?:19|20)\d{2}\b", response)
    ents = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", response)
    low = set(rt)
    uncertainty = sum(w.strip() in low for w in UNCERTAINTY) / max(1,len(rt))
    certainty = sum(w in low for w in CERTAINTY) / max(1,len(rt))
    repeated = 1 - len(set(rt))/max(1,len(rt))
    avg_len = np.mean([len(tokens(s)) for s in ss]) if ss else 0
    diversity = len(set(rt))/max(1,len(rt))
    qsim = _overlap(question, response)
    specificity = min(1.0, (len(nums)*0.08 + len(ents)*0.04 + len(rt)/120))
    contradiction = sum(1 for w in ["but","however","although","except","not"] if w in low)/5
    return {
      "word_count":len(rt), "char_count":len(response), "sentence_count":len(ss),
      "avg_sentence_length":float(avg_len), "vocabulary_diversity":float(diversity),
      "repetition_rate":float(repeated), "qa_lexical_overlap":float(qsim),
      "entity_count":len(ents), "unique_entity_count":len(set(ents)),
      "number_count":len(nums), "date_count":len(dates), "uncertainty_rate":float(uncertainty),
      "certainty_rate":float(certainty), "specificity":float(specificity),
      "contradiction_rate":float(contradiction)
    }

def signal_scores(question, response):
    f=extract_features(question,response)
    return {
      "Semantic consistency": round(100*(0.45 + 0.55*f["qa_lexical_overlap"]),1),
      "Question–answer relevance": round(100*min(1, f["qa_lexical_overlap"]*1.7 + 0.15),1),
      "Entity consistency": round(100*max(0, 1 - f["repetition_rate"] - f["contradiction_rate"]*0.3),1),
      "Linguistic uncertainty": round(100*min(1, f["uncertainty_rate"]*8 + f["certainty_rate"]*5),1),
      "Numerical consistency": round(100*max(0, 1 - min(0.7, f["number_count"]*0.04 + f["date_count"]*0.05)),1),
      "Response specificity": round(100*f["specificity"],1),
    }

def vectorize_corpus(rows):
    vec=TfidfVectorizer(ngram_range=(1,2), min_df=1, max_features=2500, stop_words="english")
    X=vec.fit_transform([r["question"]+" "+r["response"] for r in rows])
    return vec,X
