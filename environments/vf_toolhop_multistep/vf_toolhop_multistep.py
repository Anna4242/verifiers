import json
import os
from datasets import load_from_disk
import verifiers as vf

def load_environment() -> vf.Environment:
    def find_dataset_path():
        possible_paths = [
            'toolhop_verifiers_format',
            os.path.join(os.path.dirname(__file__), '../../../toolhop_verifiers_format'),
            os.path.join(os.getcwd(), 'toolhop_verifiers_format'),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"Found dataset at: {path}")
                return path
        
        raise FileNotFoundError(f"Could not find toolhop_verifiers_format dataset in any of these locations: {possible_paths}")
    
    def process_example(x):
        # Enhanced system prompt that clearly explains the expected format
        enhanced_prompt = x["prompt"].copy()
        enhanced_prompt[0]["content"] = """You are a helpful assistant that can use tools to answer multi-step reasoning questions.

For complex questions that require multiple steps:

1. First, think step-by-step inside <think>...</think> tags
2. For EACH reasoning step, use this exact format:
   <step>
   Question: [The sub-question you need to answer]
   Tool needed: [Name of the tool to use]
   </step>
3. After all steps, call the tools inside <tool>...</tool> tags as a JSON array:
   [{"name": "tool_name", "arguments": {"param": "value"}}]
4. Finally, provide your answer inside <answer>...</answer> tags

Example format:
<think>
I need to break this down into steps...
</think>

<step>
Question: What park links location A with location B?
Tool needed: geo_relationship_finder
</step>

<step>
Question: Who designed that park?
Tool needed: historical_figure_identifier
</step>

<tool>
[
  {"name": "geo_relationship_finder", "arguments": {"location_name": "Location A", "entity_types": ["park"]}},
  {"name": "historical_figure_identifier", "arguments": {"event_name": "Park Name"}}
]
</tool>

<answer>
The final answer based on tool results
</answer>"""
        
        return {
            "prompt": enhanced_prompt,
            "answer": x["answer"],
            "task": x["task"],
            "domain": x["domain"],
            "answer_type": x["answer_type"],
            "difficulty": x["difficulty"],
            "original_id": x["original_id"]
        }

    dataset_path = find_dataset_path()
    dataset = load_from_disk(dataset_path)
    
    # Apply enhanced system prompt to all examples
    def apply_enhanced_prompt(example):
        enhanced_prompt = example["prompt"].copy()
        enhanced_prompt[0]["content"] = """You are a helpful assistant that can use tools to answer multi-step reasoning questions.

For complex questions that require multiple steps:

1. First, think step-by-step inside <think>...</think> tags
2. For EACH reasoning step, use this exact format:
   <step>
   Question: [The sub-question you need to answer]
   Tool needed: [Name of the tool to use]
   </step>
3. After all steps, call the tools inside <tool>...</tool> tags as a JSON array:
   [{"name": "tool_name", "arguments": {"param": "value"}}]
4. Finally, provide your answer inside <answer>...</answer> tags

Example format:
<think>
I need to break this down into steps...
</think>

<step>
Question: What park links location A with location B?
Tool needed: geo_relationship_finder
</step>

<step>
Question: Who designed that park?
Tool needed: historical_figure_identifier
</step>

<tool>
[
  {"name": "geo_relationship_finder", "arguments": {"location_name": "Location A", "entity_types": ["park"]}},
  {"name": "historical_figure_identifier", "arguments": {"event_name": "Park Name"}}
]
</tool>

<answer>
The final answer based on tool results
</answer>"""
        
        return {
            "prompt": enhanced_prompt,
            "answer": example["answer"],
            "task": example["task"],
            "domain": example["domain"],
            "answer_type": example["answer_type"],
            "difficulty": example["difficulty"],
            "original_id": example["original_id"]
        }
    
    dataset = dataset.map(apply_enhanced_prompt, num_proc=1)

    # Parser that can handle multiple XML tags
    parser = vf.XMLParser(fields=["think", "step", "tool", "answer"], answer_field="answer")

    def check_reasoning_steps_reward_func(completion, answer):
        """Reward for identifying correct reasoning steps"""
        try:
            answer_data = json.loads(answer)
            expected_reasoning_chain = answer_data.get("reasoning_chain", [])
            
            parsed = parser.parse(completion[-1]["content"] if isinstance(completion, list) else completion)
            
            if not hasattr(parsed, 'step') or not parsed.step:
                return 0.0
            
            # Count how many expected sub-questions are mentioned in the steps
            steps_content = parsed.step.lower()
            correct_steps = 0
            
            for expected_step in expected_reasoning_chain:
                expected_question = expected_step["question"].lower()
                # Check if key words from the expected question appear in the steps
                key_words = [word for word in expected_question.split() if len(word) > 3]
                if any(key_word in steps_content for key_word in key_words):
                    correct_steps += 1
            
            if len(expected_reasoning_chain) > 0:
                return correct_steps / len(expected_reasoning_chain)
            return 0.0
            
        except Exception:
            return 0.0

    def check_tool_identification_reward_func(completion, answer):
        """Reward for identifying correct tools needed"""
        try:
            answer_data = json.loads(answer)
            expected_tool_calls = answer_data.get("tool_calls", [])
            expected_tool_names = [tool["name"] for tool in expected_tool_calls]
            
            parsed = parser.parse(completion[-1]["content"] if isinstance(completion, list) else completion)
            
            if not hasattr(parsed, 'step') or not parsed.step:
                return 0.0
            
            # Check if expected tool names are mentioned in the steps
            steps_content = parsed.step.lower()
            correct_tools_identified = 0
            
            for tool_name in expected_tool_names:
                if tool_name.lower() in steps_content:
                    correct_tools_identified += 1
            
            if len(expected_tool_names) > 0:
                return correct_tools_identified / len(expected_tool_names)
            return 0.0
            
        except Exception:
            return 0.0

    def check_tool_calls_format_reward_func(completion, answer):
        """Reward for properly formatted tool calls"""
        try:
            answer_data = json.loads(answer)
            expected_tool_calls = answer_data.get("tool_calls", [])
            
            parsed = parser.parse(completion[-1]["content"] if isinstance(completion, list) else completion)
            
            if not hasattr(parsed, 'tool') or not parsed.tool:
                return 0.0
            
            # Try to parse the tool calls as JSON
            try:
                called_tools = json.loads(parsed.tool)
                if not isinstance(called_tools, list):
                    return 0.0
                
                # Check if the structure is correct
                correctly_formatted = 0
                for tool_call in called_tools:
                    if isinstance(tool_call, dict) and "name" in tool_call and "arguments" in tool_call:
                        correctly_formatted += 1
                
                return correctly_formatted / max(1, len(called_tools))
                
            except json.JSONDecodeError:
                return 0.0
            
        except Exception:
            return 0.0

    def check_tool_calls_accuracy_reward_func(completion, answer):
        """Reward for calling the correct tools with reasonable arguments"""
        try:
            answer_data = json.loads(answer)
            expected_tool_calls = answer_data.get("tool_calls", [])
            expected_tool_names = [tool["name"] for tool in expected_tool_calls]
            
            parsed = parser.parse(completion[-1]["content"] if isinstance(completion, list) else completion)
            
            if not hasattr(parsed, 'tool') or not parsed.tool:
                return 0.0
            
            try:
                called_tools = json.loads(parsed.tool)
                if not isinstance(called_tools, list):
                    return 0.0
                
                called_tool_names = [tool.get("name", "") for tool in called_tools if isinstance(tool, dict)]
                
                # Check how many expected tools were called
                correct_tools_called = 0
                for expected_name in expected_tool_names:
                    if expected_name in called_tool_names:
                        correct_tools_called += 1
                
                return correct_tools_called / max(1, len(expected_tool_names))
                
            except json.JSONDecodeError:
                return 0.0
            
        except Exception:
            return 0.0

    def check_final_answer_reward_func(completion, answer):
        """Reward for correct final answer"""
        try:
            answer_data = json.loads(answer)
            expected_final_answer = answer_data.get("final_answer", "")
            
            parsed = parser.parse(completion[-1]["content"] if isinstance(completion, list) else completion)
            
            if not hasattr(parsed, 'answer') or not parsed.answer:
                return 0.0
            
            # Exact match gets full credit
            if parsed.answer.strip().lower() == expected_final_answer.strip().lower():
                return 1.0
            # Partial match gets partial credit
            elif expected_final_answer.lower() in parsed.answer.lower():
                return 0.7
            elif parsed.answer.lower() in expected_final_answer.lower():
                return 0.5
            else:
                return 0.0
                
        except Exception:
            return 0.0

    # Create rubric with step-by-step rewards
    rubric = vf.Rubric(funcs=[
        check_reasoning_steps_reward_func,      # 0-1: Did they identify the right reasoning steps?
        check_tool_identification_reward_func,  # 0-1: Did they identify the right tools needed?
        check_tool_calls_format_reward_func,    # 0-1: Are tool calls properly formatted?
        check_tool_calls_accuracy_reward_func,  # 0-1: Are the right tools called?
        check_final_answer_reward_func          # 0-1: Is the final answer correct?
    ])

    vf_env = vf.SingleTurnEnv(
        dataset=dataset,
        parser=parser,
        rubric=rubric,
    )
    return vf_env 