MuSiQue RAG Database System
A local RAG (Retrieval-Augmented Generation) server that simulates search engine functionality for training, based on the Jan-nano paper's approach.
Overview
This system creates a searchable database from the MuSiQue dataset using:

E5-base-v2 encoder for dense retrieval
ms-marco-MiniLM cross-encoder for reranking
FAISS for efficient similarity search
Flask server to simulate search engine API