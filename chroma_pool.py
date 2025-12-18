"""ChromaDB connection pool for reusing connections across agents."""
import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from config import EMBEDDING_MODEL, OPENAI_API_KEY, CHROMA_PERSIST_DIRECTORY, ENABLE_PROMOTIONS_AGENT

# Global singleton instances
_embeddings = None
_chroma_clients = {}


def get_embeddings():
    """Get or create singleton embeddings instance."""
    global _embeddings
    if _embeddings is None:
        _embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            api_key=OPENAI_API_KEY
        )
    return _embeddings


def get_chroma_client(collection_name: str):
    """
    Get or create ChromaDB client for specified collection.
    
    Args:
        collection_name: Name of the collection (products, knowledge, promotions)
        
    Returns:
        Chroma client instance
        
    Note:
        Promotions collection is only loaded if ENABLE_PROMOTIONS_AGENT=True in config.py
    """
    global _chroma_clients
    
    # Skip promotions if disabled
    if collection_name == "promotions" and not ENABLE_PROMOTIONS_AGENT:
        raise ValueError(f"Promotions agent is disabled. Set ENABLE_PROMOTIONS_AGENT=True in config.py to enable.")
    
    if collection_name not in _chroma_clients:
        embeddings = get_embeddings()
        _chroma_clients[collection_name] = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=CHROMA_PERSIST_DIRECTORY
        )
    return _chroma_clients[collection_name]


def clear_chroma_pool():
    """Clear all cached ChromaDB connections (useful for testing/cleanup)."""
    global _chroma_clients, _embeddings
    _chroma_clients.clear()
    _embeddings = None
