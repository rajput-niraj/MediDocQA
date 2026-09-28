# MediDocQA

Welcome to **MediDocQA**, a beginner-friendly Retrieval-Augmented Generation (RAG) project designed for answering questions over medical documents.

This project focuses on simplicity, readability, and modularity. It leverages open and free-tier tools to demonstrate the RAG lifecycle from ingestion to generation.

## Project Architecture
- **document_loader.py**: Ingests and chunks medical text/PDF documents.
- **vector_store.py**: Embeds text using `sentence-transformers` and stores them in a local `FAISS` index.
- **rag_pipeline.py**: Formats retrieved context and user queries into a prompt, generating an answer using the `Groq` API (`llama-3.1-8b-instant`).
- **config.py**: Centralized settings for chunking and models.
- **main.py**: The interactive CLI entry point.

## Setup Instructions

### 1. Prerequisites
Ensure you have Python 3.9+ installed. 

### 2. Install Dependencies
Run the following command to install the required Python packages:
```bash
pip install -r requirements.txt
```

### 3. Environment Variables
1. Copy the example environment file:
```bash
cp .env.example .env
```
2. Open `.env` and add your **Groq API Key**. You can get one for free at [console.groq.com](https://console.groq.com/).

### 4. Run the Application
Start the CLI application:
```bash
python main.py
```

On the first run, it will automatically generate a sample `sample_patient_record.txt` file and index it. You can then start asking questions like:
- *What was the patient diagnosed with?*
- *What medications should the patient take post-surgery?*
- *When is the follow-up appointment?*
