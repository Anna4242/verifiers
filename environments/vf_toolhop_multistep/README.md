# vf-toolhop-multistep

A verifier environment for the ToolHop multi-step reasoning dataset with enhanced step-by-step evaluation.

## Overview

This environment evaluates models on complex multi-step reasoning tasks that require tool usage. It provides detailed feedback on different aspects of the reasoning process, from identifying the right steps to properly formatting and executing tool calls.

## Features

### Enhanced System Prompt
- Clear instructions for multi-step reasoning format
- Explicit examples showing expected XML tag structure
- Step-by-step guidance for tool usage

### Multi-Level Reward Functions

1. **Reasoning Steps Reward** (0-1): Evaluates whether the model correctly identifies the sub-questions needed to solve the problem
2. **Tool Identification Reward** (0-1): Checks if the model identifies the correct tools needed for each step
3. **Tool Format Reward** (0-1): Validates that tool calls are properly formatted as JSON arrays
4. **Tool Accuracy Reward** (0-1): Verifies that the correct tools are actually called with appropriate arguments
5. **Final Answer Reward** (0-1): Checks if the final answer matches the expected result

### Expected Response Format

The model is expected to respond in this structure:

```
<think>
Step-by-step reasoning about the problem...
</think>

<step>
Question: What is the first sub-question?
Tool needed: tool_name_1
</step>

<step>
Question: What is the second sub-question?
Tool needed: tool_name_2
</step>

<tool>
[
  {"name": "tool_name_1", "arguments": {"param": "value"}},
  {"name": "tool_name_2", "arguments": {"param": "value"}}
]
</tool>

<answer>
The final answer based on reasoning and tool results
</answer>
```

## Dataset Structure

The environment expects a dataset with the following structure:
- `prompt`: Chat messages (system + user)
- `answer`: JSON string containing:
  - `reasoning_chain`: List of expected sub-questions and answers
  - `tool_calls`: List of expected tool calls with arguments
  - `final_answer`: Expected final answer
- Additional metadata: `domain`, `answer_type`, `difficulty`, etc.

## Usage

```python
import verifiers as vf
from vf_toolhop_multistep import load_environment

# Load the environment
env = load_environment()

# Run evaluation
results = env.evaluate(
    model="your-model-name",
    num_examples=10,
    rollouts_per_example=1
)
```

## Reward Function Details

### 1. Reasoning Steps Reward
- Extracts content from `<step>` tags
- Matches key words from expected sub-questions
- Returns proportion of correctly identified reasoning steps

### 2. Tool Identification Reward  
- Checks if expected tool names appear in the `<step>` sections
- Returns proportion of correctly identified tools

### 3. Tool Format Reward
- Validates JSON structure of tool calls in `<tool>` tags
- Ensures each tool call has required "name" and "arguments" fields
- Returns proportion of correctly formatted tool calls

### 4. Tool Accuracy Reward
- Compares called tools against expected tools
- Returns proportion of expected tools that were actually called

### 5. Final Answer Reward
- Exact match: 1.0 points
- Partial match (expected in actual): 0.7 points  
- Partial match (actual in expected): 0.5 points
- No match: 0.0 points

## Installation

```bash
cd verifiers/environments/vf_toolhop_multistep
pip install -e .
```

## Evaluation Command

```bash
vf-eval vf_toolhop_multistep \
  --model "your-model" \
  --num-examples 10 \
  --rollouts-per-example 1 \
  --temperature 0.3
```

This environment provides comprehensive evaluation of multi-step reasoning capabilities, helping identify specific areas where models succeed or struggle in complex tool-using scenarios. 