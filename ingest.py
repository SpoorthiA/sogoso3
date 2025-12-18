"""Data ingestion script to load JSON data into ChromaDB."""
import json
import os
from typing import List
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import (
    OPENAI_API_KEY,
    CHROMA_PERSIST_DIRECTORY,
    KNOWLEDGE_FILE,
    PRODUCTS_FILE,
    PROMOTIONS_FILE,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP
)


def load_knowledge_data() -> List[Document]:
    """Load and chunk knowledge base data."""
    print("Loading knowledge data...")
    with open(KNOWLEDGE_FILE, 'r') as f:
        data = json.load(f)
    
    documents = []
    for entry in data['sop_entries']:
        # Create comprehensive text representation
        content = f"""
Policy ID: {entry['policy_id']}
Title: {entry['title']}
Applies To: {', '.join(entry['applies_to'])}

Policy:
{entry['policy_text']}

Exceptions:
{chr(10).join('- ' + exc for exc in entry['exceptions'])}

Examples:
{chr(10).join('- ' + ex for ex in entry['examples'])}
        """.strip()
        
        doc = Document(
            page_content=content,
            metadata={
                "policy_id": entry['policy_id'],
                "title": entry['title'],
                "applies_to": ", ".join(entry['applies_to']),  # Convert list to string
                "source": "knowledge"
            }
        )
        documents.append(doc)
    
    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunked_docs = text_splitter.split_documents(documents)
    print(f"Created {len(chunked_docs)} knowledge chunks from {len(documents)} policies")
    return chunked_docs


def load_product_data() -> List[Document]:
    """Load product data with metadata for filtering."""
    print("Loading product data...")
    with open(PRODUCTS_FILE, 'r') as f:
        data = json.load(f)
    
    documents = []
    for product in data['products']:
        # Use vector_text as main content
        content = product.get('vector_text', product['name'])
        
        # Create comprehensive metadata for filtering
        metadata = {
            "id": product['id'],
            "name": product['name'],
            "category": product['category'],
            "brand": product['brand'],
            "price": product['price'],
            "stock": product['stock'],
            "source": "product"
        }
        
        # Add optional fields
        if 'material' in product:
            metadata['material'] = product['material']
        if 'model' in product:
            metadata['model'] = product['model']
        if 'size' in product:
            metadata['size'] = product['size']
        if 'weight' in product:
            metadata['weight'] = product['weight']
        if 'length' in product:
            metadata['length'] = product['length']
        
        doc = Document(
            page_content=content,
            metadata=metadata
        )
        documents.append(doc)
    
    print(f"Created {len(documents)} product documents")
    return documents


def load_promotion_data() -> List[Document]:
    """Load and chunk promotion data."""
    print("Loading promotion data...")
    with open(PROMOTIONS_FILE, 'r') as f:
        data = json.load(f)
    
    documents = []
    for promo in data['promotions']:
        # Create detailed text representation
        content = f"""
Promotion: {promo['name']} (ID: {promo['promo_id']})
Type: {promo['type']}
Discount: {promo.get('discount_value', 'N/A')}
Categories: {', '.join(promo['applicable_categories'])}
Conditions: {', '.join(promo['conditions'])}
Valid: {promo['validity']}
        """.strip()
        
        doc = Document(
            page_content=content,
            metadata={
                "promo_id": promo['promo_id'],
                "name": promo['name'],
                "type": promo['type'],
                "applicable_categories": ", ".join(promo['applicable_categories']),  # Convert list to string
                "validity": promo['validity'],
                "source": "promotion"
            }
        )
        documents.append(doc)
    
    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunked_docs = text_splitter.split_documents(documents)
    print(f"Created {len(chunked_docs)} promotion chunks from {len(documents)} promotions")
    return chunked_docs


def ingest_data():
    """Main ingestion function."""
    print("=" * 50)
    print("SOGOSO Data Ingestion Pipeline")
    print("=" * 50)
    
    # Initialize embeddings
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=OPENAI_API_KEY
    )
    
    # Create persist directory
    os.makedirs(CHROMA_PERSIST_DIRECTORY, exist_ok=True)
    
    # 1. Ingest Knowledge Base
    print("\n[1/3] Processing Knowledge Base...")
    knowledge_docs = load_knowledge_data()
    knowledge_db = Chroma.from_documents(
        documents=knowledge_docs,
        embedding=embeddings,
        collection_name="knowledge",
        persist_directory=CHROMA_PERSIST_DIRECTORY
    )
    print(f"✓ Knowledge base ingested: {knowledge_db._collection.count()} documents")
    
    # 2. Ingest Products
    print("\n[2/3] Processing Products...")
    product_docs = load_product_data()
    product_db = Chroma.from_documents(
        documents=product_docs,
        embedding=embeddings,
        collection_name="products",
        persist_directory=CHROMA_PERSIST_DIRECTORY
    )
    print(f"✓ Products ingested: {product_db._collection.count()} documents")
    
    # 3. Ingest Promotions
    print("\n[3/3] Processing Promotions...")
    promotion_docs = load_promotion_data()
    promotion_db = Chroma.from_documents(
        documents=promotion_docs,
        embedding=embeddings,
        collection_name="promotions",
        persist_directory=CHROMA_PERSIST_DIRECTORY
    )
    print(f"✓ Promotions ingested: {promotion_db._collection.count()} documents")
    
    print("\n" + "=" * 50)
    print("✓ Data Ingestion Complete!")
    print("=" * 50)
    print(f"ChromaDB location: {CHROMA_PERSIST_DIRECTORY}")
    print(f"Collections: knowledge, products, promotions")


if __name__ == "__main__":
    ingest_data()
