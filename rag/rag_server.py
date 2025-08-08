#!/usr/bin/env python3
"""
Local RAG server that simulates search engine functionality for training.
Based on Jan-nano paper's approach.
"""

import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
from flask import Flask, request, jsonify
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np

app = Flask(__name__)

class RAGServer:
    """RAG server that provides search and document retrieval."""
    
    def __init__(self, index_path: str = "musique_index"):
        self.index_path = Path(index_path)
        
        print("Loading RAG database...")
        self.load_index()
        
        print("Loading encoders...")
        self.encoder = SentenceTransformer('intfloat/e5-base-v2')
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
        
        print(f"RAG server ready with {len(self.documents)} documents")
    
    def load_index(self):
        """Load the FAISS index and documents from disk."""
        # Load FAISS index
        index_file = self.index_path / "faiss.index"
        self.index = faiss.read_index(str(index_file))
        
        # Load documents
        docs_file = self.index_path / "documents.pkl"
        with open(docs_file, 'rb') as f:
            self.documents = pickle.load(f)
        
        # Create ID to document mapping
        self.doc_id_map = {doc['id']: doc for doc in self.documents}
    
    def search(self, query: str, top_k_retrieval: int = 15, top_k_rerank: int = 10) -> List[Dict[str, Any]]:
        """
        Two-stage retrieval as described in the paper:
        1. Dense retrieval with E5-base-v2 (top 15)
        2. Cross-encoder reranking with ms-marco-MiniLM (top 10)
        """
        # Stage 1: Dense retrieval
        query_text = f"query: {query}"
        query_embedding = self.encoder.encode([query_text], convert_to_numpy=True)
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k_retrieval)
        
        # Get documents
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.documents):
                doc = self.documents[idx].copy()
                doc['retrieval_score'] = float(score)
                results.append(doc)
        
        # Stage 2: Cross-encoder reranking
        if results:
            pairs = [[query, f"{doc['title']} {doc['text']}"] for doc in results]
            rerank_scores = self.cross_encoder.predict(pairs)
            
            for doc, score in zip(results, rerank_scores):
                doc['rerank_score'] = float(score)
            
            # Sort by reranking score
            results.sort(key=lambda x: x['rerank_score'], reverse=True)
            results = results[:top_k_rerank]
        
        return results
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get full document by ID."""
        # Handle both numeric doc_N format and original IDs
        if doc_id.startswith('doc_'):
            try:
                idx = int(doc_id.replace('doc_', ''))
                if 0 <= idx < len(self.documents):
                    return self.documents[idx]
            except ValueError:
                pass
        
        return self.doc_id_map.get(doc_id)

# Global server instance
rag_server = None

@app.route('/retrieve', methods=['POST'])
def retrieve():
    """
    Endpoint for search retrieval.
    Expects JSON: {
        "queries": ["query1", ...],
        "topk_retrieval": 15,
        "topk_rerank": 10,
        "return_scores": false
    }
    """
    global rag_server
    
    data = request.json
    queries = data.get('queries', [])
    topk_retrieval = data.get('topk_retrieval', 15)
    topk_rerank = data.get('topk_rerank', 10)
    
    all_results = []
    
    for query in queries:
        results = rag_server.search(query, topk_retrieval, topk_rerank)
        
        # Format results as expected by training code
        formatted_results = []
        for i, doc in enumerate(results):
            # Create preview (first 150 characters)
            preview = doc['text'][:150]
            if len(doc['text']) > 150:
                preview += '...'
            
            formatted_results.append({
                'doc_id': f"doc_{doc['doc_id']}",
                'title': doc['title'],
                'text': preview,  # Preview for search results
                'score': doc.get('rerank_score', 0.0)
            })
        
        all_results.append(formatted_results)
    
    return jsonify({'result': all_results})

@app.route('/visit', methods=['POST'])
def visit():
    """
    Endpoint for retrieving full document content.
    Expects JSON: {"url": "doc_id"}
    """
    global rag_server
    
    data = request.json
    doc_id = data.get('url', '')
    
    doc = rag_server.get_document(doc_id)
    
    if doc:
        result = [{
            'doc_id': doc_id,
            'title': doc['title'],
            'text': doc['text']  # Full text for visit
        }]
    else:
        result = [{
            'doc_id': doc_id,
            'title': 'Not Found',
            'text': f'Document {doc_id} not found'
        }]
    
    return jsonify({'result': [result]})

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'documents': len(rag_server.documents)})

def main():
    """Start the RAG server."""
    import argparse
    
    parser = argparse.ArgumentParser(description='RAG server for training')
    parser.add_argument('--index-path', type=str, default='musique_index',
                        help='Path to the index')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host to run server on')
    parser.add_argument('--port', type=int, default=2223,
                        help='Port to run server on')
    args = parser.parse_args()
    
    global rag_server
    rag_server = RAGServer(index_path=args.index_path)
    
    print(f"\nStarting RAG server on {args.host}:{args.port}")
    print(f"Endpoints:")
    print(f"  - POST /retrieve - Search for documents")
    print(f"  - POST /visit - Get full document content")
    print(f"  - GET /health - Health check")
    
    app.run(host=args.host, port=args.port, debug=False)

if __name__ == '__main__':
    main()