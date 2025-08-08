# vf-musique-rag

MuSiQue RAG Environment for multi-hop reasoning with retrieval-augmented generation.

## Overview

This environment integrates the MuSiQue dataset with RAG capabilities for training models on complex multi-hop reasoning tasks. It provides tools for searching a knowledge base and retrieving specific documents to answer questions that require connecting multiple pieces of information.

## Features

- **Multi-hop reasoning**: Designed for questions requiring multiple steps of reasoning
- **RAG integration**: Connects to a RAG server for document retrieval
- **Custom tools**: Specialized search and document retrieval functions
- **ToolEnv integration**: Uses Verifiers' native tool-calling capabilities

## Tools

1. **search_musique**: Search the MuSiQue knowledge base
   - Args: `query` (str), `num_results` (int, default=10)
   
2. **visit_document**: Retrieve full content of a specific document
   - Args: `doc_id` (str)
   
3. **multi_hop_search**: Two-step search for complex questions
   - Args: `initial_query` (str), `follow_up_query` (str)

## Setup

1. **Install the environment**:
```bash
cd verifiers/environments/vf_musique_rag
pip install -e .
```

2. **Start your RAG server** (assumes you have a compatible RAG server):
```bash
# Example with FlashRAG or similar
python rag_server.py --index-path musique_index --port 2223
```

3. **Set environment variables**:
```bash
export MUSIQUE_RAG_SERVER="http://localhost:2223"
```

## Usage

### Basic Usage

```python
import verifiers as vf

# Load the environment
env = vf.load_environment("vf-musique-rag")

# Use with OpenAI client for evaluation
from openai import OpenAI
client = OpenAI()

results = env.evaluate(
    client=client, 
    model="gpt-4", 
    num_examples=100
)
```

### Training

```python
import verifiers as vf

# Load environment and model
env = vf.load_environment("vf-musique-rag")
model, tokenizer = vf.get_model_and_tokenizer("Qwen/Qwen2.5-7B-Instruct")

# Configure training
args = vf.grpo_defaults(run_name="musique-rag-training")
args.max_steps = 500
args.per_device_train_batch_size = 2

# Create trainer
trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=env,
    args=args,
)

# Start training
trainer.train()
```

## Dataset

Uses the MuSiQue dataset from HuggingFace: `dgslibisey/MuSiQue`

- **Train split**: 19.9k examples
- **Validation split**: 2.42k examples
- **Task type**: Multi-hop question answering
- **Format**: Questions with decomposed reasoning steps

## Expected Response Format

The environment expects responses in this XML format:

```xml
<think>
Step-by-step reasoning about the question...
</think>

<tool>
{"name": "search_musique", "args": {"query": "search terms", "num_results": 5}}
</tool>

<result>
Search results will appear here...
</result>

<tool>
{"name": "visit_document", "args": {"doc_id": "doc_123"}}
</tool>

<result>
Document content will appear here...
</result>

<answer>
Final answer based on retrieved information
</answer>
```

## Requirements

- Verifiers framework
- RAG server running on specified port
- Internet connection for dataset download
- Python 3.8+

## Troubleshooting

1. **RAG server connection issues**: Ensure your RAG server is running and accessible
2. **Dataset loading errors**: Check internet connection and HuggingFace access
3. **Tool execution errors**: Verify RAG server API compatibility

## License

MIT License 