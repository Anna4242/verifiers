import json
import os
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from datasets import load_from_disk
from tqdm import tqdm
import pickle

# Try to import faiss-gpu, fall back to faiss-cpu
try:
    import faiss
    gpu_available = faiss.get_num_gpus() > 0
    print(f"FAISS imported successfully. GPU available: {gpu_available}")
except ImportError:
    print("FAISS not found. Please install faiss-cpu or faiss-gpu")
    raise

class MuSiQueRAGDatabase:
    """
    Build a RAG database from MuSiQue dataset as described in Jan-nano paper.
    Uses E5-base-v2 for dense retrieval and ms-marco-MiniLM for reranking.
    """
    
    def __init__(self, 
                 index_path: str = "musique_index",
                 use_gpu: bool = None):
        self.index_path = Path(index_path)
        self.index_path.mkdir(exist_ok=True)
        
        # Auto-detect GPU if not specified
        if use_gpu is None:
            use_gpu = faiss.get_num_gpus() > 0
        
        self.use_gpu = use_gpu and faiss.get_num_gpus() > 0
        
        # Initialize encoders as per paper
        print("Loading E5-base-v2 encoder for dense retrieval...")
        self.encoder = SentenceTransformer('intfloat/e5-base-v2')
        
        print("Loading ms-marco-MiniLM cross-encoder for reranking...")
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
        
        self.documents = []
        self.index = None
        
        print(f"Using {'GPU' if self.use_gpu else 'CPU'} for FAISS index")
        
    def extract_documents_from_musique(self, dataset_path: str) -> List[Dict[str, Any]]:
        """Extract all unique documents from MuSiQue dataset."""
        print(f"Loading dataset from {dataset_path}...")
        dataset = load_from_disk(dataset_path)
        
        documents = []
        seen_ids = set()
        
        # Process both train and validation splits
        for split_name in ['train', 'validation']:
            if split_name not in dataset:
                continue
                
            split = dataset[split_name]
            print(f"Processing {split_name} split with {len(split)} examples...")
            
            for example in tqdm(split, desc=f"Extracting from {split_name}"):
                # MuSiQue has 'paragraphs' field with supporting documents
                if 'paragraphs' in example and example['paragraphs']:
                    for para in example['paragraphs']:
                        if isinstance(para, dict):
                            # Extract paragraph info
                            para_id = para.get('idx', para.get('id', ''))
                            para_title = para.get('title', '')
                            para_text = para.get('paragraph_text', para.get('text', ''))
                            
                            # Create unique ID
                            doc_id = f"{para_title}_{para_id}" if para_id else para_title
                            
                            if doc_id and doc_id not in seen_ids and para_text:
                                seen_ids.add(doc_id)
                                documents.append({
                                    'id': doc_id,
                                    'title': para_title,
                                    'text': para_text,
                                    'doc_id': len(documents)  # Numeric ID for FAISS
                                })
        
        print(f"Extracted {len(documents)} unique documents")
        return documents
    
    def build_index(self, documents: List[Dict[str, Any]]):
        """Build FAISS index."""
        self.documents = documents
        
        # Prepare texts for encoding (E5 requires "passage: " prefix)
        texts = [f"passage: {doc['title']} {doc['text']}" for doc in documents]
        
        print(f"Encoding {len(texts)} documents with E5-base-v2...")
        print("This may take a while for large datasets...")
        
        # Encode in batches to avoid memory issues
        batch_size = 32
        embeddings_list = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding batches"):
            batch = texts[i:i+batch_size]
            batch_embeddings = self.encoder.encode(
                batch,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            embeddings_list.append(batch_embeddings)
        
        embeddings = np.vstack(embeddings_list)
        
        # Normalize embeddings for cosine similarity
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        print(f"Building FAISS index with dimension {dimension}...")
        
        if self.use_gpu:
            try:
                print("Attempting to use GPU for FAISS index...")
                res = faiss.StandardGpuResources()
                index_flat = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
                self.index = faiss.index_cpu_to_gpu(res, 0, index_flat)
            except Exception as e:
                print(f"GPU initialization failed: {e}")
                print("Falling back to CPU...")
                self.use_gpu = False
                self.index = faiss.IndexFlatIP(dimension)
        else:
            print("Using CPU for FAISS index...")
            self.index = faiss.IndexFlatIP(dimension)
        
        # Add embeddings to index
        print("Adding vectors to index...")
        self.index.add(embeddings.astype('float32'))
        print(f"Added {self.index.ntotal} vectors to index")
        
    def save(self):
        """Save the index and documents to disk."""
        # Save FAISS index
        index_file = self.index_path / "faiss.index"
        print(f"Saving FAISS index to {index_file}...")
        
        if self.use_gpu:
            # Transfer GPU index to CPU for saving
            cpu_index = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(cpu_index, str(index_file))
        else:
            faiss.write_index(self.index, str(index_file))
        
        # Save documents
        docs_file = self.index_path / "documents.pkl"
        print(f"Saving {len(self.documents)} documents to {docs_file}...")
        with open(docs_file, 'wb') as f:
            pickle.dump(self.documents, f)
        
        # Save metadata
        metadata = {
            'num_documents': len(self.documents),
            'encoder_model': 'intfloat/e5-base-v2',
            'cross_encoder_model': 'cross-encoder/ms-marco-MiniLM-L-12-v2',
            'use_gpu': self.use_gpu
        }
        metadata_file = self.index_path / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Database saved to {self.index_path}")
        
    def search(self, query: str, top_k: int = 15) -> List[Dict[str, Any]]:
        """Search for relevant documents using dense retrieval."""
        # Encode query (E5 requires "query: " prefix)
        query_text = f"query: {query}"
        query_embedding = self.encoder.encode([query_text], convert_to_numpy=True)
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        
        # Search in FAISS
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.documents):
                doc = self.documents[idx].copy()
                doc['score'] = float(score)
                results.append(doc)
        
        return results
    
    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = 10) -> List[Dict[str, Any]]:
        """Rerank documents using cross-encoder."""
        if not documents:
            return []
        
        # Prepare pairs for cross-encoder
        pairs = [[query, f"{doc['title']} {doc['text']}"] for doc in documents]
        
        # Get reranking scores
        scores = self.cross_encoder.predict(pairs)
        
        # Add scores and sort
        for doc, score in zip(documents, scores):
            doc['rerank_score'] = float(score)
        
        # Sort by reranking score
        documents.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        return documents[:top_k]

def main():
    """Build the RAG database from MuSiQue dataset."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Build RAG database from MuSiQue dataset')
    parser.add_argument('--dataset-path', type=str, default='../musique_dataset',
                        help='Path to MuSiQue dataset')
    parser.add_argument('--index-path', type=str, default='musique_index',
                        help='Path to save the index')
    parser.add_argument('--use-cpu', action='store_true',
                        help='Force CPU usage even if GPU is available')
    args = parser.parse_args()
    
    # Initialize database builder
    db = MuSiQueRAGDatabase(
        index_path=args.index_path,
        use_gpu=not args.use_cpu
    )
    
    # Extract documents from dataset
    documents = db.extract_documents_from_musique(args.dataset_path)
    
    if not documents:
        print("No documents found in dataset!")
        return
    
    # Build index
    db.build_index(documents)
    
    # Save everything
    db.save()
    
    # Test search
    print("\n=== Testing search functionality ===")
    test_query = "When was McDonald's founded"
    results = db.search(test_query, top_k=15)
    reranked = db.rerank(test_query, results, top_k=10)
    
    print(f"\nQuery: {test_query}")
    print(f"Found {len(results)} initial results, reranked to {len(reranked)}")
    if reranked:
        print(f"Top result: {reranked[0]['title'][:50]}...")

if __name__ == "__main__":
    main()