#!/usr/bin/env python3
"""Test the RAG database and server functionality."""

import requests
import json
from build_rag_index import MuSiQueRAGDatabase

def test_database():
    """Test direct database search."""
    print("=== Testing Database ===")
    
    db = MuSiQueRAGDatabase(index_path="musique_index")
    
    # Load existing index
    from pathlib import Path
    import pickle
    import faiss
    
    db.index = faiss.read_index(str(Path("musique_index") / "faiss.index"))
    with open(Path("musique_index") / "documents.pkl", 'rb') as f:
        db.documents = pickle.load(f)
    
    # Test search
    query = "What is the capital of France"
    results = db.search(query, top_k=5)
    
    print(f"\nQuery: {query}")
    print(f"Found {len(results)} results")
    for i, doc in enumerate(results[:3]):
        print(f"\n{i+1}. {doc['title']}")
        print(f"   {doc['text'][:100]}...")
        print(f"   Score: {doc['score']:.4f}")

def test_server():
    """Test the RAG server endpoints."""
    print("\n=== Testing Server ===")
    print("Make sure the server is running: python rag_server.py")
    
    base_url = "http://localhost:2223"
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health")
        print(f"\nHealth check: {response.json()}")
    except:
        print("Server not running! Start it with: python rag_server.py")
        return
    
    # Test search
    search_payload = {
        "queries": ["When was the Declaration of Independence signed"],
        "topk_retrieval": 15,
        "topk_rerank": 10,
        "return_scores": False
    }
    
    response = requests.post(f"{base_url}/retrieve", json=search_payload)
    results = response.json()['result'][0]
    
    print(f"\nSearch results for: {search_payload['queries'][0]}")
    for i, doc in enumerate(results[:3]):
        print(f"\n{i+1}. {doc['title']}")
        print(f"   ID: {doc['doc_id']}")
        print(f"   Preview: {doc['text']}")
    
    # Test visit
    if results:
        doc_id = results[0]['doc_id']
        visit_payload = {"url": doc_id}
        
        response = requests.post(f"{base_url}/visit", json=visit_payload)
        full_doc = response.json()['result'][0][0]
        
        print(f"\nFull document for {doc_id}:")
        print(f"Title: {full_doc['title']}")
        print(f"Text: {full_doc['text'][:200]}...")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'server':
        test_server()
    else:
        test_database()
        print("\n\nTo test server, run: python test_rag.py server")