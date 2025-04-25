
🏥 Hospital Readmission Prediction System with LLMs

📌 Overview

Hospital readmissions are a major concern due to their economic and clinical burden. This project introduces a state-of-the-art Hospital Readmission Prediction System leveraging Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG). Using MIMIC-III electronic health records, we fine-tune multiple LLMs (ClinicalBERT, Mistral 7B, LLama 2/3, GatorTron) to predict readmissions within 30 days post-discharge. The system provides accurate, interpretable predictions and clinical decision support through a Django-based UI.

🔍 Problem Statement

Hospital readmissions drive up healthcare costs and are often preventable. Traditional models underperform with unstructured data like clinical notes. This system aims to harness LLMs and NLP to analyze unstructured EHR data and predict patient readmission risk.

🧠 Solution Architecture



Key Components:

Data Preprocessing: MIMIC-III Admissions and NoteEvents tables merged and cleaned

Knowledge Base: Clinical notes embedded using transformer encoders into vector DB (FAISS/ChromaDB)

Retriever: Finds relevant patient context for each query

Generator (LLM): Predicts readmission risk using the retrieved context

UI Layer: Django web app for clinicians to interact with the model

🔨 Tech Stack

Language Models: ClinicalBERT, Mistral 7B, LLama2, GatorTron

Frameworks: HuggingFace Transformers, PyTorch, TensorFlow

UI & Backend: Django, Python

Storage: Google Drive (data), FAISS / ChromaDB (embeddings)

Cloud: AWS EC2/GPU (training), PhysioNet (data source)

📈 Evaluation Metrics

Model Metrics: Accuracy, F1-score, AUROC, AUPRC

RAG Evaluation: BLEU, ROUGE scores for text generation

Real-time Inference: Time-to-response, contextual accuracy

📊 Data Source

MIMIC-III Dataset

50,000+ ICU patient records

Admissions + clinical notes (NoteEvents)

Preprocessed using Python scripts and stored in cloud folders

✅ Features

Early detection of high-risk patients

Explainable predictions with text highlights

Custom care plan suggestions

Secure, scalable design using free-tier tools (Google Drive, GitHub, etc.)

🚀 Future Scope

Multi-modal integration (e.g., lab tests + clinical notes)

Integration with hospital EMR systems

Deployment via Docker or Streamlit

Expansion to cover chronic disease prediction

