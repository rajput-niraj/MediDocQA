import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_medical_document(file_path: str) -> str:
    """
    Reads a medical document (.txt or .pdf) and returns raw cleaned text.
    
    Args:
        file_path (str): The absolute or relative path to the document.
        
    Returns:
        str: The extracted and cleaned text from the document.
        
    RAG Lifecycle:
        Step 1: Ingestion. We need to extract raw text from files before we can process it.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
        
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
            
    elif ext == '.pdf':
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        return text
        
    else:
        raise ValueError(f"Unsupported file format: {ext}. Please provide a .txt or .pdf file.")


def chunk_text(raw_text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    Splits raw text into manageable chunks with overlap.
    
    Why chunk overlap matters in RAG:
    Sentences or paragraphs might be split right in the middle of an important concept.
    Overlap ensures that context is preserved across chunk boundaries, preventing
    the loss of crucial medical information that spans across the cut.
    
    Args:
        raw_text (str): The full document text.
        chunk_size (int): Maximum number of characters in a chunk.
        chunk_overlap (int): Number of characters to overlap between consecutive chunks.
        
    Returns:
        list[str]: A list of text chunks.
        
    RAG Lifecycle:
        Step 2: Chunking. We break large documents into smaller pieces so they fit 
        into the LLM's context window and improve the relevance of search results.
    """
    # We use RecursiveCharacterTextSplitter as it tries to split on paragraphs/sentences first
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    chunks = text_splitter.split_text(raw_text)
    return chunks
