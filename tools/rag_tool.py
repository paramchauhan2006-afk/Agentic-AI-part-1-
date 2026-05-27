import os
import glob
import numpy as np
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the Gemini API client
# google-genai client will automatically use GEMINI_API_KEY from environment
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else genai.Client()

EMBEDDING_MODEL = "gemini-embedding-001"

def load_and_chunk_documents(knowledge_base_dir: str = "knowledge_base", chunk_size: int = 500, overlap: int = 100) -> list[dict]:
    """
    Reads plain text/markdown files from the local knowledge_base directory
    and splits them into overlapping text chunks.
    
    Returns a list of dicts: [{'text': chunk_text, 'source': file_name, 'chunk_idx': idx}]
    """
    chunks = []
    if not os.path.exists(knowledge_base_dir):
        print(f"Warning: Directory '{knowledge_base_dir}' not found.")
        return chunks

    # Find all txt and md files
    file_patterns = [os.path.join(knowledge_base_dir, "*.txt"), os.path.join(knowledge_base_dir, "*.md")]
    files = []
    for pattern in file_patterns:
        files.extend(glob.glob(pattern))

    for file_path in files:
        file_name = os.path.basename(file_path)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

        if not content.strip():
            continue

        # Perform character-based chunking with overlap
        start = 0
        chunk_idx = 0
        while start < len(content):
            end = start + chunk_size
            chunk_text = content[start:end].strip()
            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "source": file_name,
                    "chunk_idx": chunk_idx
                })
                chunk_idx += 1
            # Move the window forward
            start += (chunk_size - overlap)

    return chunks

def generate_embeddings(texts: list[str], is_query: bool = False) -> list[list[float]]:
    """
    Generates text embeddings using the official google-genai SDK.
    """
    if not texts:
        return []

    task_type = "RETRIEVAL_QUERY" if is_query else "RETRIEVAL_DOCUMENT"
    
    try:
        # Batch call to embed_content
        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=texts,
            config=types.EmbedContentConfig(
                task_type=task_type
            )
        )
        # Extract and return raw float lists
        return [emb.values for emb in response.embeddings]
    except Exception as e:
        print(f"Error generating embeddings via google-genai: {e}")
        # Return dummy embeddings (zeros) in case of API failure for local execution fallback
        # Real applications should handle this error appropriately
        raise e

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Computes the cosine similarity between two vector lists.
    """
    vec_a = np.array(a)
    vec_b = np.array(b)
    dot = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))

def retrieve_top_k(query: str, knowledge_base_dir: str = "knowledge_base", k: int = 3) -> list[dict]:
    """
    Performs vector similarity search to return the top-K chunks matching the query.
    """
    # 1. Load and chunk documents
    chunks = load_and_chunk_documents(knowledge_base_dir)
    if not chunks:
        return []

    # 2. Extract texts to embed
    texts_to_embed = [chunk["text"] for chunk in chunks]

    # 3. Generate embeddings in batch for document chunks, and single embedding for query
    chunk_embeddings = generate_embeddings(texts_to_embed, is_query=False)
    query_embedding = generate_embeddings([query], is_query=True)[0]

    # 4. Calculate similarities
    results = []
    for chunk, emb in zip(chunks, chunk_embeddings):
        sim = cosine_similarity(query_embedding, emb)
        results.append({
            "text": chunk["text"],
            "source": chunk["source"],
            "similarity": sim
        })

    # 5. Sort by similarity descending and select top K
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:k]
