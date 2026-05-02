# Hallucination Detection System

The **Hallucination Detection System** is an end-to-end NLP framework designed to assess the factual reliability of AI-generated content. As Large Language Models (LLMs) scale, they often produce *hallucinations* — fluent but inaccurate information. This system provides an automated pipeline to analyze, verify, and classify outputs to ensure data integrity.

---

## Key Features

- **Automated Verification Pipeline:**  
  Structured workflow involving text preprocessing, TF-IDF feature extraction, and Jaccard similarity evaluation.

- **Knowledge Retrieval Integration:**  
  Real-time fetching of ground-truth evidence via the Wikipedia API.

- **Ambiguity Resolution:**  
  Intelligent disambiguation module to clarify multi-faceted topics (e.g., "Python" the snake vs. "Python" the language).

- **Hybrid Analysis Logic:**  
  Combines statistical analysis with Named Entity Recognition (NER) to catch mismatches in dates, names, and core claims.

- **Transparent Reasoning:**  
  Visual step-by-step breakdown of the detection logic and verdict.

- **Interactive Dashboard:**  
  Flask-based web UI featuring semantic heatmaps, confidence scores, and exportable PDF audit reports.

---

## Technical Stack

- **Language:** Python 3.x  
- **Backend:** Flask  
- **Machine Learning:** Scikit-learn (Logistic Regression, TF-IDF)  
- **NLP Libraries:** NLTK, spaCy (NER, Lemmatization), Transformers (NLI)  
- **Data Handling:** Pandas, NumPy  
- **Frontend:** HTML5, CSS3, JavaScript (Jinja2)  

---

## System Architecture

The system operates across four specialized layers:

1. **Preprocessing Layer:**  
   Standardizes data via tokenization, stopword removal, and script validation.

2. **Semantic Layer:**  
   Captures context using n-grams and embeddings (Word2Vec, FastText, or GloVe).

3. **Classification Layer:**  
   A Logistic Regression model determines the logical relationship (entailment vs. contradiction) between the AI output and retrieved evidence.

4. **Decision Engine:**  
   Aggregates semantic scores and entity mismatches to assign a status:  
   - Not Hallucinated  
   - Partially Verified  
   - Hallucinated

---
