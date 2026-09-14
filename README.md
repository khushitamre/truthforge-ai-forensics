# TRUTHFORGE
## AI Integrity & Hallucination Forensics Engine

TRUTHFORGE is a local, recruiter-ready NLP and machine-learning application for interrogating AI-generated responses before they become decisions. It estimates hallucination-like risk from linguistic, semantic, entity, and numerical patterns, then explains the signals that contributed to the result.

> TRUTHFORGE estimates hallucination risk from learned linguistic and semantic patterns. It does not guarantee factual correctness.

## Product surface

| Module | Purpose |
|---|---|
| Home | Product overview and operating principles |
| Forensic Analysis | Question/response analysis, risk gauge, signals, claims, and PDF report |
| Case Files | Local SQLite history and case reopening |
| Model Lab | Training metrics, model rationale, and limitations |
| System Info | Pipeline, stack, scope, and future work |

## Architecture

The pipeline is: raw question and response → cleaning → sentence segmentation → claim extraction → engineered linguistic features → TF-IDF representation → Logistic Regression → transparent signal blend → claim-level findings → SQLite case file and ReportLab PDF.

Version 1 intentionally excludes RAG, vector databases, external LLM APIs, and web verification. The design therefore remains understandable and reproducible by a data scientist.

## Features

The feature layer calculates word and character counts, sentence count, average sentence length, vocabulary diversity, repetition rate, question–answer lexical overlap, entity counts, numerical and date mentions, uncertainty and certainty indicators, contradiction markers, and response specificity. These are accompanied by claim-level predictions and top contributing signals.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

## Model development

`train_model.py` is the reproducible training entry point. It saves `models/final_model.pkl` and `models/metrics.pkl`. The current seed corpus is documented in `data/README.md`; it is an offline development baseline and should be replaced with a larger licensed benchmark for production.

## Limitations and future work

Pattern-based classification cannot prove a claim false without reliable evidence. False positives, false negatives, distribution shift, domain mismatch, and calibration limitations are expected. Future versions may add external evidence retrieval, RAG, source credibility scoring, human verification, knowledge graphs, and model monitoring.

## Interview discussion points

The project supports discussion of feature engineering, class weighting, precision versus recall, explainability, claim segmentation, SQLite persistence, PDF reporting, and responsible AI limitations. The most important design decision is semantic honesty: the product reports a risk signal rather than a truth score.

## License

Use this repository as an engineering prototype. Add an explicit license and dataset terms before public redistribution.

[1]: https://github.com/ys-zong/ HaluEval and hallucination-evaluation research resources
[2]: https://scikit-learn.org/stable/modules/linear_model.html "Scikit-learn linear models"
