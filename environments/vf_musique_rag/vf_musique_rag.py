
import json
import os
import requests
import random
from typing import Any, Dict, List, Optional, Tuple, Callable
from datasets import load_dataset

import verifiers as vf
from verifiers import (
    Messages,
    State,
    XMLParser,
)
from verifiers.envs.multiturn_env import MultiTurnEnv

# ==================== MuSiQue XML Tools ====================

def search(keywords: str, max_results: Optional[int] = 10, region: Optional[str] = "wt-wt") -> list:
    """
    This function performs a keyword-based search on the MuSiQue knowledge base.
    
    Args:
        keywords: Search keywords string
        max_results: Maximum number of results to return (default: 10)
        region: Search region (default: "wt-wt" for worldwide)
        
    Returns:
        List of search results with title, doc_id, and preview text
    """
    server_url = os.environ.get("MUSIQUE_RAG_SERVER", "http://localhost:2223")
    
    # Simulate probabilistic request failures
    failure_types = [
        "503 Server Error",
        "429 Too Many Requests", 
        "403 Forbidden",
        "ConnectTimeout",
        "ReadTimeout",
        "ConnectionError"
    ]
    
    # 15% chance of failure to simulate real-world conditions
    if random.random() < 0.15:
        error_type = random.choice(failure_types)
        return [{"error": f"Search failed: {error_type}. You may retry or try different keywords."}]
    
    try:
        payload = {
            "queries": [keywords],
            "topk_retrieval": min(max_results * 2, 30),
            "topk_rerank": max_results,
            "return_scores": False
        }
        
        response = requests.post(
            f"{server_url}/retrieve",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code != 200:
            return [{"error": f"Search failed with status {response.status_code}"}]
        
        result = response.json()
        documents = result.get('result', [[]])[0]
        
        if not documents:
            return [{"message": "No results found for your keywords."}]
        
        # Format results as list of dictionaries
        search_results = []
        for i, doc in enumerate(documents[:max_results], 1):
            doc_id = doc.get('doc_id', f'doc_{i}')
            title = doc.get('title', f'Document {i}')
            text = doc.get('text', '').strip()
            
            # Truncate preview if too long
            preview = text[:200] + '...' if len(text) > 200 else text
            
            search_results.append({
                "title": title,
                "doc_id": doc_id,
                "preview": preview,
                "url": f"doc://{doc_id}"
            })
        
        return search_results
        
    except requests.exceptions.RequestException as e:
        return [{"error": f"Search request failed: {str(e)}"}]
    except Exception as e:
        return [{"error": f"Search error: {str(e)}"}]


def fetch(url: str, mode: str = "raw") -> str:
    """
    This function retrieves the content from a specified document URL.
    
    Args:
        url: Document URL to retrieve (e.g., "doc://doc_123")
        mode: Retrieval mode - "raw", "markdown", or "truncate" (default: "raw")
        
    Returns:
        Document content in the specified format
    """
    server_url = os.environ.get("MUSIQUE_RAG_SERVER", "http://localhost:2223")
    
    # Simulate probabilistic request failures
    failure_types = [
        "503 Server Error",
        "429 Too Many Requests",
        "403 Forbidden", 
        "ConnectTimeout",
        "ReadTimeout",
        "ConnectionError"
    ]
    
    # 10% chance of failure
    if random.random() < 0.10:
        error_type = random.choice(failure_types)
        return f"Fetch failed: {error_type}. You may retry this request or select an alternative URL."
    
    try:
        # Extract doc_id from URL (format: doc://doc_123)
        if url.startswith("doc://"):
            doc_id = url[6:]  # Remove "doc://" prefix
        else:
            doc_id = url
            
        payload = {"url": doc_id}
        
        response = requests.post(
            f"{server_url}/visit",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code != 200:
            return f"Error: Could not retrieve document {doc_id} (status: {response.status_code})"
        
        result = response.json()
        documents = result.get('result', [[]])[0]
        
        if not documents:
            return f"Error: Document {doc_id} not found"
        
        doc = documents[0]
        title = doc.get('title', 'Untitled')
        content = doc.get('text', '').strip()
        
        # Apply different modes
        if mode == "markdown":
            # Convert to markdown-like format
            formatted_content = f"# {title}\n\n{content}"
            formatted_content = formatted_content.replace('\n\n', '\n\n---\n\n')
            return formatted_content
            
        elif mode == "truncate":
            # Clean and extract essential content, optimize for tokens
            lines = content.split('\n')
            essential_lines = []
            for line in lines:
                line = line.strip()
                if line and len(line) > 10:
                    essential_lines.append(line)
            
            truncated = ' '.join(essential_lines)
            if len(truncated) > 1000:
                truncated = truncated[:1000] + '...'
                
            return f"Document: {title}\n\nContent: {truncated}"
            
        else:  # raw mode (default)
            return f"Document: {title}\nID: {doc_id}\n\nContent:\n{content}"
        
    except Exception as e:
        return f"Error retrieving document {url}: {str(e)}"


# ==================== Custom XML Tool Environment ====================

class MuSiQueXMLEnv(MultiTurnEnv):
    """MuSiQue XML environment using XML tool format."""
    
    def __init__(
        self,
        tools: List[Callable] = [],
        system_prompt: str = "",
        parser: XMLParser = XMLParser(fields=["think", ("tool", "answer")]),
        env_parser: XMLParser = XMLParser(fields=["result"]),
        max_turns: int = 10,
        **kwargs,
    ):
        self.tools = {tool.__name__: tool for tool in tools}
        self.parser = parser
        self.env_parser = env_parser
        
        super().__init__(
            system_prompt=system_prompt,
            parser=parser,
            max_turns=max_turns,
            **kwargs,
        )

    def is_completed(self, messages: Messages, state: State, **kwargs: Any) -> bool:
        """Check if the conversation is completed (has an answer)."""
        return self.parser.parse_answer(messages) is not None

    def call_tool(self, tool_json: str, max_chars: int = 1024, **kwargs) -> str:
        """Call a tool based on JSON command."""
        try:
            command = json.loads(tool_json)
            if not isinstance(command, dict):
                return 'Error: Tool command must be a JSON object, e.g. \'{"name": "tool_name", "args": {"arg1": "value1", "arg2": "value2"}}\''

            tool_name = command.get("name")
            if not tool_name:
                return 'Error: Tool command must specify \'name\', e.g. \'{"name": "tool_name", "args": {"arg1": "value1", "arg2": "value2"}}\''

            if tool_name not in self.tools:
                return (
                    f"Error: Unknown tool '{tool_name}'. Available tools: {list(self.tools.keys())}. "
                    + 'Please format your tool call as \'{"name": "tool_name", "args": {"arg1": "value1", "arg2": "value2"}}\''
                )

            tool_func = self.tools[tool_name]
            tool_args = command.get("args", {})
            if isinstance(tool_args, str):
                return f"Error: Arguments for {tool_name} must be a JSON object, not a string."

            # Call the tool function with arguments
            result = tool_func(**tool_args)
            if max_chars > 0 and len(str(result)) > max_chars:
                result = str(result)[:max_chars] + "..."
            return str(result)
        except Exception as e:
            return (
                f"Error: {str(e)}. "
                + 'Please format your tool call as \'{{"name": "tool_name", "args": {{"arg1": "value1", "arg2": "value2"}}}}\''
            )

    def env_response(
        self, messages: Messages, state: State, **kwargs
    ) -> Tuple[Messages, State]:
        """Process tool calls and return results."""
        try:
            parsed = self.parser.parse(messages[-1]["content"])  # type: ignore
            # Check if we got a valid tool field (not just None from failed parsing)
            if hasattr(parsed, "tool") and parsed.tool is not None:
                result = self.call_tool(parsed.tool)
                if len(result.strip()) > 0:
                    return [
                        {
                            "role": "user",
                            "content": self.env_parser.format(result=result),
                        }
                    ], state
                else:
                    return [
                        {
                            "role": "user",
                            "content": "Error: Tool execution returned empty output.",
                        }
                    ], state
        except Exception as e:
            print(f"Error in env_response: {e}")
        return [
            {
                "role": "user",
                "content": "Error: Tool command not found or invalid XML format. Please ensure correct formatting.",
            }
        ], state

# ==================== MuSiQue System Prompt ====================

MUSIQUE_SYSTEM_PROMPT = """You are an AI assistant with access to the MuSiQue knowledge base - a comprehensive collection of facts for answering complex, multi-hop reasoning questions.

Your task is to answer questions that often require connecting information from multiple sources. Use the available tools to search and retrieve relevant information.

Available tools:
1. **search(keywords, max_results=10, region="wt-wt")**: Search the knowledge base using keywords
2. **fetch(url, mode="raw")**: Retrieve full content from a document URL
   - mode options: "raw" (full text), "markdown" (formatted), "truncate" (optimized)

Important guidelines:
- Start with brief thinking about what to search for (<think>...</think>) - keep it concise (1-2 sentences)
- Use search to find relevant documents first
- Use fetch to get detailed content from promising documents
- For multi-hop questions, you may need multiple searches and fetches
- Provide your final answer in <answer>...</answer> tags
- If tools fail (network errors, timeouts), retry or try alternative searches

Tool call format:
<tool>
{"name": "search", "args": {"keywords": "your search terms", "max_results": 5}}
</tool>

<tool>
{"name": "fetch", "args": {"url": "doc://doc_id", "mode": "raw"}}
</tool>

The system will respond with results in <result>...</result> tags.

Example:
<think>
I need to find which institution owns "The Collegian" newspaper, then find when that institution was founded.
</think>

<tool>
{"name": "search", "args": {"keywords": "The Collegian newspaper owner institution", "max_results": 5}}
</tool>

Remember: Keep thinking brief, search strategically, and connect information from multiple sources when needed.
"""

# ==================== Environment Loader ====================

def load_environment(
    rag_server_url: str = "http://localhost:2223",
    max_turns: int = 10,
    **kwargs
) -> vf.Environment:
    """
    Load the MuSiQue XML environment.
    
    Args:
        rag_server_url: URL of the RAG server (default: http://localhost:2223)
        max_turns: Maximum conversation turns (default: 10)
        **kwargs: Additional arguments passed to the environment
    
    Returns:
        vf.Environment: Configured MuSiQue environment
    """
    os.environ["MUSIQUE_RAG_SERVER"] = rag_server_url
    dataset = load_dataset("dgslibisey/MuSiQue", split="train")
    
    def process_example(example):
        return {
            "question": example["question"],
            "answer": example["answer"],
            "task": "musique-xml",
            "paragraphs": example["paragraphs"],
            "decomposition": example.get("decomposition", [])
        }
    
    dataset = dataset.map(process_example, remove_columns=dataset.column_names)
    
    tools = [search, fetch]
    
    # Create parser
    parser = XMLParser(fields=["think", ("tool", "answer")])
    env_parser = XMLParser(fields=["result"])
    
    # Use simple rubric - just check if answer exists
    def answer_exists_reward_func(completion, answer, **kwargs) -> float:
        try:
            final_answer = parser.parse_answer(completion)
            return 1.0 if final_answer else 0.0
        except:
            return 0.0
    
    rubric = vf.Rubric(funcs=[answer_exists_reward_func])
    
    vf_env = MuSiQueXMLEnv(
        dataset=dataset,
        system_prompt=MUSIQUE_SYSTEM_PROMPT,
        tools=tools,
        parser=parser,
        env_parser=env_parser,
        rubric=rubric,
        max_turns=max_turns,
        **kwargs
    )
    
    return vf_env

__all__ = ["load_environment", "search", "fetch", "MuSiQueXMLEnv"]
