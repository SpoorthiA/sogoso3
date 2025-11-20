"""Data ingestion script for NEW data format (GCP DataStore export)."""
import json
import os
from typing import List
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from config import (
    OPENAI_API_KEY,
    CHROMA_PERSIST_DIRECTORY,
    KNOWLEDGE_FILE,
    PRODUCTS_FILE,
    EMBEDDING_MODEL,
    ENABLE_PROMOTIONS_AGENT
)


def load_knowledge_data() -> List[Document]:
    """
    Load knowledge data from NEW format.
    
    NEW FORMAT:
    {
        "chunk_id": "...",
        "score": 0.x,
        "content": "text content...",
        "metadata": {
            "uri": "gs://...",
            "title": "..."
        }
    }
    """
    print("Loading knowledge data (NEW FORMAT)...")
    with open(KNOWLEDGE_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    documents = []
    for entry in data:
        content = entry.get('content', '')
        
        # Skip empty content
        if not content or not content.strip():
            continue
        
        metadata = {
            "chunk_id": entry.get('chunk_id', ''),
            "title": entry.get('metadata', {}).get('title', 'Unknown'),
            "uri": entry.get('metadata', {}).get('uri', ''),
            "source": "knowledge"
        }
        
        doc = Document(
            page_content=content,
            metadata=metadata
        )
        documents.append(doc)
    
    print(f"✓ Created {len(documents)} knowledge documents")
    return documents


def load_product_data() -> List[Document]:
    """
    Load product data from NEW format.
    
    NEW FORMAT:
    {
        "name": "projects/.../chunks/c1",
        "id": "c1",
        "content": {
            "category": "...",
            "product_name": "...",
            "product_description": "...",
            "brand": "...",
            "price": 123.45,
            "attributes": [...],
            "product_specifications": [...],
            ...
        },
        "document_metadata": {
            "uri": "gs://...",
            "title": "..."
        }
    }
    """
    print("Loading product data (NEW FORMAT)...")
    with open(PRODUCTS_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    documents = []
    for product in data:
        content_obj = product.get('content', {})
        
        # Skip if content is missing or not a dict
        if not isinstance(content_obj, dict):
            continue
        
        # Build comprehensive text content
        product_name = content_obj.get('product_name', 'Unknown Product')
        product_desc = content_obj.get('product_description', '')
        brand = content_obj.get('brand', '')
        category = content_obj.get('category', '')
        price = content_obj.get('price', 0)
        
        # Build text representation
        content_parts = [
            f"Product: {product_name}",
            f"Brand: {brand}" if brand else "",
            f"Category: {category}" if category else "",
            f"Price: ${price}" if price else "",
            f"\n{product_desc}" if product_desc else ""
        ]
        
        # Add additional details if available
        additional = content_obj.get('product_additional_details', '')
        highlights = content_obj.get('product_highlights', '')
        info = content_obj.get('product_information', '')
        
        if highlights:
            content_parts.append(f"\nHighlights: {highlights}")
        if info:
            content_parts.append(f"\nInformation: {info}")
        if additional:
            content_parts.append(f"\nDetails: {additional}")
        
        content = "\n".join(filter(None, content_parts))
        
        # Extract metadata
        doc_metadata = product.get('document_metadata', {})
        
        metadata = {
            "id": product.get('id', ''),
            "product_name": product_name,
            "brand": brand,
            "category": category,
            "price": str(price) if price else "0",
            "title": doc_metadata.get('title', product_name),
            "uri": doc_metadata.get('uri', ''),
            "source": "product"
        }
        
        # Add review info - always include, even if 0
        rating = content_obj.get('review_rating', 0)
        review_count = content_obj.get('review_count', 0)
        metadata['review_rating'] = str(rating) if rating is not None else "0"
        metadata['review_count'] = str(int(review_count)) if review_count is not None else "0"
        
        doc = Document(
            page_content=content,
            metadata=metadata
        )
        documents.append(doc)
    
    print(f"✓ Created {len(documents)} product documents")
    return documents


def ingest_data():
    """Main ingestion function for NEW data format."""
    print("=" * 60)
    print("SOGOSO Data Ingestion Pipeline (NEW FORMAT)")
    print("=" * 60)
    
    # Initialize embeddings
    print(f"\nInitializing embeddings model: {EMBEDDING_MODEL}")
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=OPENAI_API_KEY
    )
    
    # Create persist directory
    os.makedirs(CHROMA_PERSIST_DIRECTORY, exist_ok=True)
    
    # 1. Ingest Knowledge Base
    print("\n[1/2] Processing Knowledge Base...")
    knowledge_docs = load_knowledge_data()
    if knowledge_docs:
        knowledge_db = Chroma.from_documents(
            documents=knowledge_docs,
            embedding=embeddings,
            collection_name="knowledge",
            persist_directory=CHROMA_PERSIST_DIRECTORY
        )
        print(f"✓ Knowledge base ingested: {knowledge_db._collection.count()} documents")
    else:
        print("⚠ No knowledge documents to ingest")
    
    # 2. Ingest Products
    print("\n[2/2] Processing Products...")
    product_docs = load_product_data()
    if product_docs:
        product_db = Chroma.from_documents(
            documents=product_docs,
            embedding=embeddings,
            collection_name="products",
            persist_directory=CHROMA_PERSIST_DIRECTORY
        )
        print(f"✓ Products ingested: {product_db._collection.count()} documents")
    else:
        print("⚠ No product documents to ingest")
    
    # 3. Skip promotions (not available in new dataset)
    if ENABLE_PROMOTIONS_AGENT:
        print("\n⚠ WARNING: ENABLE_PROMOTIONS_AGENT=True but promotions.json not found")
        print("  Set ENABLE_PROMOTIONS_AGENT=False in config.py")
    
    print("\n" + "=" * 60)
    print("✓ Data Ingestion Complete!")
    print("=" * 60)
    print(f"ChromaDB location: {CHROMA_PERSIST_DIRECTORY}")
    print(f"Collections created: knowledge, products")
    print(f"Total documents: {len(knowledge_docs) + len(product_docs)}")


if __name__ == "__main__":
    ingest_data()
