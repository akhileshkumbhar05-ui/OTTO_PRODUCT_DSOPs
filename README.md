# OTTO Product DS-Ops — Session Recommender (End-to-End + Deployment)

This project is an end-to-end **Product Data Science + DS-Ops** pipeline built on the OTTO session dataset format (`train.jsonl`).  
It demonstrates a production-style workflow:

- **Ingest** local JSONL session logs
- **Build** a memory-safe feature store (Parquet)
- **Generate** bounded candidates via co-visitation + popularity
- **Train** a lightweight ranking scorer (CPU-safe)
- **Evaluate** Recall@K / HitRate@K
- **Deploy** an interactive Streamlit app (Local, Docker, EC2)

---

## What this project does

### Session Recommendation
Given a user’s session event sequence (aids), the system recommends top next items using:
1. **Candidate generation (bounded)**  
   - Co-visitation neighbors (within-session sliding window)  
   - Global popularity fallback  
2. **Lightweight scoring model**  
   - Logistic model trained on sampled sessions  
   - Features: popularity, co-visitation strength, recency, repetition

This avoids heavy learning-to-rank pipelines that can crash a laptop.

---

## Requirements
- Python 3.10+ recommended (tested with Python 3.11)
- Local dataset: `data/raw/train.jsonl`

---

## Setup (Windows PowerShell)
```powershell
cd otto-product-dsops
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
