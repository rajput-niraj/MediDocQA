import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL_NAME

def get_embedding_model() -> SentenceTransformer:
    """
    Loads the local sentence-transformer model for generating embeddings.
    
    Returns:
        SentenceTransformer: The loaded HuggingFace embedding model.
        
    RAG Lifecycle:
        Step 3: Embedding (Initialization). We need a model to convert text chunks 
        and user queries into numerical vectors that capture their semantic meaning.
    """
    print(f"Loading embedding model '{EMBEDDING_MODEL_NAME}'...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return model


def build_faiss_index(chunks: list[str]) -> tuple[faiss.Index, list[str]]:
    """
    Embeds chunks, indexes them in a local FAISS IndexFlatL2, and returns the index.
    
    Args:
        chunks (list[str]): The list of text chunks to index.
        
    Returns:
        tuple[faiss.Index, list[str]]: The built FAISS index and the corresponding chunks.
        
    RAG Lifecycle:
        Step 3 & 4: Embedding & Vector Storage. We convert chunks to vectors 
        and store them in a FAISS index for fast nearest-neighbor search.
    """
    model = get_embedding_model()
    
    print(f"Embedding {len(chunks)} chunks...")
    # Generate embeddings. The output is a numpy array of shape (num_chunks, embedding_dim)
    embeddings = model.encode(chunks)
    
    # FAISS expects float32
    embeddings = np.array(embeddings).astype('float32')
    
    # Get the dimension of the embeddings
    dimension = embeddings.shape[1]
    
    print(f"Building FAISS index with dimension {dimension}...")
    # IndexFlatL2 performs exact search using Euclidean distance
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    return index, chunks


def search_faiss_index(query: str, index: faiss.Index, chunks: list[str], top_k: int = 3) -> list[str]:
    """
    Embeds the query, runs nearest-neighbor search, and returns the top matching text snippets.
    
    Args:
        query (str): The user's question.
        index (faiss.Index): The populated FAISS index.
        chunks (list[str]): The list of original text chunks corresponding to the index.
        top_k (int): Number of top results to return.
        
    Returns:
        list[str]: The top_k most relevant text chunks.
        
    RAG Lifecycle:
        Step 5: Retrieval. We convert the user's question into a vector and find 
        the most similar vectors in our store, retrieving the original text chunks.
    """
    model = get_embedding_model()
    
    # Embed the query
    query_embedding = model.encode([query]).astype('float32')
    
    # Perform the search
    distances, indices = index.search(query_embedding, top_k)
    
    # Retrieve the corresponding text chunks
    retrieved_chunks = []
    for idx in indices[0]:
        if idx != -1 and idx < len(chunks):
            retrieved_chunks.append(chunks[idx])
            
    return retrieved_chunks


# =====================================================================
# Optional: Pinecone Integration (Commented out for learning purposes)
# =====================================================================
# def init_pinecone_index(chunks: list[str], pinecone_api_key: str, index_name: str):
#     """
#     Example of how to use Pinecone instead of FAISS.
#     
#     Requires `pip install pinecone-client`
#     """
#     # from pinecone import Pinecone, ServerlessSpec
#     # pc = Pinecone(api_key=pinecone_api_key)
#     # 
#     # # Create index if it doesn't exist
#     # if index_name not in pc.list_indexes().names():
#     #     pc.create_index(
#     #         name=index_name,
#     #         dimension=384, # For 'all-MiniLM-L6-v2'
#     #         metric='euclidean',
#     #         spec=ServerlessSpec(cloud='aws', region='us-east-1')
#     #     )
#     # 
#     # index = pc.Index(index_name)
#     # model = get_embedding_model()
#     # embeddings = model.encode(chunks)
#     # 
#     # # Prepare vectors for upsert: [(id, vector, metadata), ...]
#     # vectors = [(str(i), emb.tolist(), {"text": chunk}) for i, (emb, chunk) in enumerate(zip(embeddings, chunks))]
#     # index.upsert(vectors=vectors)
#     # return index
