"""
Global configuration settings for MediDocQA.
"""

# Chunking parameters
# Size of each text chunk (in characters)
CHUNK_SIZE = 500
# Overlap between consecutive chunks to maintain context
CHUNK_OVERLAP = 50

# Embedding model
# We use a lightweight, fast, and open-source embedding model from HuggingFace
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'

# LLM parameters
# We use Groq API with an open-weights model for fast generation
GROQ_MODEL_NAME = 'llama-3.1-8b-instant'

# Retrieval parameters
# Number of top relevant documents to retrieve for context
TOP_K_DOCUMENTS = 3
