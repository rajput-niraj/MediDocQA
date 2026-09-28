import os
from groq import Groq
from config import GROQ_MODEL_NAME, TOP_K_DOCUMENTS
from vector_store import search_faiss_index
import faiss

def format_prompt(context_chunks: list[str], user_query: str) -> str:
    """
    Assembles retrieved text snippets into a grounded prompt with guardrails.
    
    Args:
        context_chunks (list[str]): The retrieved text chunks from the document.
        user_query (str): The user's question.
        
    Returns:
        str: The fully formatted prompt to send to the LLM.
        
    RAG Lifecycle:
        Step 6 (Prep): Prompt Formatting. We combine the retrieved context and the 
        user's query, adding strict instructions to prevent hallucinations and 
        ensure a medical disclaimer is provided.
    """
    context_text = "\n\n---\n\n".join(context_chunks)
    
    prompt = f"""You are a helpful and knowledgeable medical assistant AI. 
    You have been provided with the following medical document excerpts as context.
    
    CONTEXT:
    {context_text}
    
    USER QUESTION:
    {user_query}
    
    INSTRUCTIONS:
    1. Answer the user's question strictly based on the provided CONTEXT.
    2. If the answer is not contained within the CONTEXT, explicitly state that you do not have enough information to answer. Do not guess or hallucinate.
    3. Always include the following disclaimer at the end of your answer: "DISCLAIMER: I am an AI, not a doctor. This information is for educational purposes and should not be used as medical advice. Always consult a healthcare professional."
    """
    return prompt


def generate_answer_with_groq(prompt: str) -> str:
    """
    Calls Groq's Chat Completion API with the formatted prompt and returns the output.
    
    Args:
        prompt (str): The fully formatted prompt containing context and instructions.
        
    Returns:
        str: The LLM's generated answer.
        
    RAG Lifecycle:
        Step 6: Generation. We send our highly contextualized prompt to the LLM 
        and receive a grounded answer.
    """
    api_key = os.getenv("GROQ_API_KEY")

    client = Groq(api_key=api_key)
    
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model= "openai/gpt-oss-120b",
        temperature=0.0, # 0.0 for more factual, less creative responses
    )
    
    return response.choices[0].message.content


def ask_medidoc(query: str, index: faiss.Index, chunks: list[str]) -> str:
    """
    The end-to-end coordinator: takes a user question, retrieves chunks, 
    formats the prompt, and fetches the answer.
    
    Args:
        query (str): The user's question.
        index (faiss.Index): The populated FAISS index.
        chunks (list[str]): The corresponding text chunks.
        
    Returns:
        str: The final answer to display to the user.
        
    RAG Lifecycle:
        End-to-End Orchestration (Retrieval -> Prompting -> Generation).
    """
    # 1. Retrieve
    retrieved_chunks = search_faiss_index(query, index, chunks, top_k=TOP_K_DOCUMENTS)
    
    if not retrieved_chunks:
        return "I could not find any relevant information in the loaded documents."
        
    # 2. Format Prompt
    prompt = format_prompt(retrieved_chunks, query)
    
    # 3. Generate Answer
    answer = generate_answer_with_groq(prompt)
    
    return answer
