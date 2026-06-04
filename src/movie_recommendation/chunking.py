"""
Text chunking strategies for movie descriptions
"""
import re
import numpy as np
from sentence_transformers import SentenceTransformer


def fixed_size_chunks(text: str, chunk_size: int = 100, overlap_percent: float = 0.15) -> list[str]:
    """
    Split text into fixed-size chunks based on word count
    
    Args:
        text: Input text to chunk
        chunk_size: Number of words per chunk
        overlap_percent: Percentage of overlap between chunks
    
    Returns:
        List of text chunks
    """
    words = text.split()
    chunks: list[str] = []
    
    overlap = int(chunk_size * overlap_percent)
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i+chunk_size]
        if chunk_words:
            chunks.append(' '.join(chunk_words))
    
    return chunks


def sentence_chunks(text: str, max_sentences: int = 3) -> list[str]:
    """
    Split text into chunks based on sentences
    
    Args:
        text: Input text to chunk
        max_sentences: Maximum sentences per chunk
    
    Returns:
        List of text chunks
    """
    sentences: list[str] = re.split(r'[.?!]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    chunks: list[str] = []
    
    for i in range(0, len(sentences), max_sentences):
        chunk_sentences = sentences[i:i + max_sentences]
        if chunk_sentences:
            chunks.append('. '.join(chunk_sentences))
    
    return chunks


def semantic_chunks(text: str, similarity_threshold: float = 0.5) -> list[str]:
    """
    Split text into chunks based on semantic similarity between sentences
    
    Args:
        text: Input text to chunk
        similarity_threshold: Similarity threshold for grouping sentences
    
    Returns:
        List of semantically coherent chunks
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')
    sentences: list[str] = re.split(r'[.]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return []
    
    embeddings = model.encode(sentences)
    
    chunks = []
    current_chunk: list[str] = [sentences[0]]
    
    for i in range(1, len(sentences)):
        similarity = np.dot(embeddings[i - 1], embeddings[i]) / (
            np.linalg.norm(embeddings[i-1]) * np.linalg.norm(embeddings[i])
        )
        
        if similarity < similarity_threshold:
            chunks.append('. '.join(current_chunk))
            current_chunk = [sentences[i]]
        else:
            current_chunk.append(sentences[i])
    
    if current_chunk:
        chunks.append('. '.join(current_chunk))
    
    return chunks