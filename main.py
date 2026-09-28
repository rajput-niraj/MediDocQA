import os
from dotenv import load_dotenv

from config import CHUNK_SIZE, CHUNK_OVERLAP
from document_loader import load_medical_document, chunk_text
from vector_store import build_faiss_index
from rag_pipeline import ask_medidoc

def create_dummy_medical_text(file_path: str):
    """Creates a sample dummy medical document if one doesn't exist."""
    dummy_text = """
Patient Discharge Summary
Patient Name: John Doe
Date of Admission: 2026-09-15
Date of Discharge: 2026-09-20
Diagnosis: Acute appendicitis, successfully treated with laparoscopic appendectomy.

Treatment Course:
The patient presented to the ER on Sept 15 with severe lower right quadrant abdominal pain, fever, and nausea. An ultrasound confirmed acute appendicitis. An emergency laparoscopic appendectomy was performed on the same day. The surgery was uncomplicated. Post-operatively, the patient was started on IV antibiotics (Ceftriaxone and Metronidazole) for 24 hours, then transitioned to oral Amoxicillin-Clavulanate for 5 days.

Discharge Instructions:
- Rest for 1 week. Avoid heavy lifting (over 10 lbs) for 4 weeks.
- Keep the incision sites clean and dry. You may shower, but do not take a bath or swim for 2 weeks.
- Take pain medication (Acetaminophen 500mg) as needed every 6 hours.
- Follow up with Dr. Smith in 2 weeks for a post-operative check and suture removal if necessary.
- Return to the ER immediately if you experience severe pain, fever over 101°F, or redness/pus at the incision sites.
"""
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(dummy_text.strip())
    print(f"Created sample medical document at {file_path}")

def main():
    # Load environment variables from .env
    load_dotenv()
    
    print("Welcome to MediDocQA - Your RAG Medical Assistant!")
    print("-" * 50)
    
    # Setup sample data
    sample_doc_path = "Documents/file.pdf"
    if not os.path.exists(sample_doc_path):
        create_dummy_medical_text(sample_doc_path)
        
    print(f"\n--- INGESTION & CHUNKING ---")
    raw_text = load_medical_document(sample_doc_path)
    chunks = chunk_text(raw_text, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"Document chunked into {len(chunks)} pieces.")
    
    print(f"\n--- EMBEDDING & VECTOR STORE ---")
    index, chunks = build_faiss_index(chunks)
    print("FAISS Index built successfully.")
    
    print("\n--- READY ---")
    print("Type your medical questions below based on the sample document.")
    print("Type 'exit' or 'quit' to stop.\n")
    
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
            
        if not user_input.strip():
            continue
            
        print("MediDocQA is thinking...")
        try:
            answer = ask_medidoc(user_input, index, chunks)
            print(f"\nMediDocQA:\n{answer}")
        except Exception as e:
            print(f"\n[Error]: {e}")

if __name__ == "__main__":
    main()
