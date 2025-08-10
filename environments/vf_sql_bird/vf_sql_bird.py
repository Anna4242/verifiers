"""
SQL BIRD Environment for Verifiers Framework
=============================================

This environment enables training models on SQL generation using the BIRD dataset
with GRPO (Group Relative Policy Optimization).

Dataset Structure:
- Creates Hugging Face Dataset with:
  * 'prompt': Complete database schema + evidence + question
  * 'answer': Gold standard SQL query (string)
  * 'info': Evaluation metadata (dict) containing db_id, difficulty, etc.

Scoring System:
- 0.0: Non-executable SQL queries
- 0.1: Executable SQL queries
- 1.0: Exact match and executable queries

Directory structure expected:
/workspace/anushka/train/bird_train/
├── train.json          # Training queries and expected SQL
├── train_databases/    # SQLite database files
├── train_gold.sql      # Gold standard SQL queries
└── train_tables.json   # Table schemas

Installation:
1. Save this file as: environments/vf_sql_bird/vf_sql_bird.py
2. Create pyproject.toml in the same directory
3. Install: vf-install sql_bird -p ./environments
"""

import json
import os
import sqlite3
import re
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import tempfile
import shutil

from datasets import Dataset
from openai import AsyncOpenAI, OpenAI

import verifiers as vf
from verifiers.envs.multiturn_env import MultiTurnEnv
from verifiers.types import Messages, State
from verifiers.parsers.xml_parser import XMLParser
from verifiers.rubrics.rubric import Rubric


class SQLExecutor:
    """Executes SQL queries against SQLite databases safely."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.temp_db_path = None
        
    def __enter__(self):
        """Create a temporary copy of the database for safe execution."""
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.temp_db_path = self.temp_db.name
        self.temp_db.close()
        shutil.copy2(self.db_path, self.temp_db_path)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up temporary database."""
        if self.temp_db_path and os.path.exists(self.temp_db_path):
            os.unlink(self.temp_db_path)
            
    def execute(self, query: str, timeout: float = 5.0) -> Tuple[bool, Any, str]:
        """
        Execute SQL query with timeout and error handling.
        
        Returns:
            Tuple of (success: bool, result: Any, error_msg: str)
        """
        if not self.temp_db_path:
            return False, None, "Database not initialized"
            
        try:
            conn = sqlite3.connect(self.temp_db_path, timeout=timeout)
            conn.execute("PRAGMA query_only = ON")  # Read-only mode
            cursor = conn.cursor()
            
            # Execute query
            cursor.execute(query)
            
            # Fetch results based on query type
            query_lower = query.lower().strip()
            if query_lower.startswith('select'):
                result = cursor.fetchall()
            else:
                result = cursor.rowcount
                
            conn.close()
            return True, result, ""
            
        except sqlite3.Error as e:
            return False, None, f"SQL Error: {str(e)}"
        except Exception as e:
            return False, None, f"Execution Error: {str(e)}"


class SQLBirdEnv(MultiTurnEnv):
    """
    Environment for SQL query generation using the BIRD dataset.
    
    Creates a Hugging Face Dataset with:
    - 'prompt': Complete database schema + evidence + question
    - 'answer': Gold standard SQL query (string) 
    - 'info': Evaluation metadata (dict)
    
    Scoring system:
    - 0.0: Non-executable SQL queries
    - 0.1: Executable SQL queries  
    - 1.0: Exact match and executable queries
    
    Supports both single-turn and multi-turn interactions with SQL refinement.
    """
    
    def __init__(
        self,
        bird_data_path: str = "/workspace/anushka/train/bird_train",
        max_turns: int = 3,
        allow_refinement: bool = True,
        use_execution_feedback: bool = True,
        system_prompt: str = None,
        parser: XMLParser = None,
        rubric: Rubric = None,
        **kwargs
    ):
        self.bird_data_path = Path(bird_data_path)
        self.databases_path = self.bird_data_path / "train_databases"
        self.allow_refinement = allow_refinement
        self.use_execution_feedback = use_execution_feedback
        
        # Load BIRD dataset
        self.bird_data = self._load_bird_data()
        
        # Convert to HF dataset format
        dataset = self._create_dataset()
        
        # Setup parser
        if parser is None:
            parser = XMLParser(
                fields=["reasoning", "sql", "explanation"],
                answer_field="sql"
            )
        
        # Setup system prompt
        if system_prompt is None:
            system_prompt = self._get_default_system_prompt()
            
        # Setup rubric
        if rubric is None:
            rubric = self._create_default_rubric(parser)
            
        super().__init__(
            dataset=dataset,
            system_prompt=system_prompt,
            parser=parser,
            rubric=rubric,
            max_turns=max_turns,
            message_type="chat",
            **kwargs
        )
        
    def _load_bird_data(self) -> List[Dict]:
        """Load BIRD training data."""
        train_file = self.bird_data_path / "train.json"
        with open(train_file, 'r') as f:
            data = json.load(f)
        return data
        
    def _load_table_schema(self, db_id: str) -> str:
        """Load and format comprehensive table schema for a database."""
        tables_file = self.bird_data_path / "train_tables.json"
        with open(tables_file, 'r') as f:
            tables_data = json.load(f)
            
        for db_info in tables_data:
            if db_info['db_id'] == db_id:
                schema_str = f"Database: {db_id}\n"
                schema_str += "=" * (len(db_id) + 10) + "\n\n"
                
                # Get table names
                table_names = db_info.get('table_names', [])
                column_names = db_info.get('column_names', [])
                column_types = db_info.get('column_types', [])
                foreign_keys = db_info.get('foreign_keys', [])
                primary_keys = db_info.get('primary_keys', [])
                
                # Build comprehensive schema
                for table_idx, table_name in enumerate(table_names):
                    schema_str += f"Table: {table_name}\n"
                    schema_str += "-" * (len(table_name) + 7) + "\n"
                    
                    # Get columns for this table
                    table_columns = []
                    for col_idx, col_info in enumerate(column_names):
                        if col_info[0] == table_idx:
                            col_name = col_info[1]
                            col_type = column_types[col_idx] if col_idx < len(column_types) else "TEXT"
                            
                            # Check if primary key
                            is_pk = col_idx in primary_keys
                            pk_marker = " (PK)" if is_pk else ""
                            
                            # Check for foreign keys
                            fk_info = ""
                            for fk in foreign_keys:
                                if fk[0] == col_idx:
                                    ref_table_idx = column_names[fk[1]][0]
                                    ref_table = table_names[ref_table_idx] if ref_table_idx < len(table_names) else "unknown"
                                    ref_col = column_names[fk[1]][1]
                                    fk_info = f" -> {ref_table}.{ref_col}"
                                    break
                            
                            table_columns.append(f"  - {col_name} ({col_type}){pk_marker}{fk_info}")
                    
                    if table_columns:
                        schema_str += "Columns:\n"
                        schema_str += "\n".join(table_columns)
                    else:
                        schema_str += "No columns found"
                    
                    schema_str += "\n\n"
                
                # Add foreign key relationships summary
                if foreign_keys:
                    schema_str += "Foreign Key Relationships:\n"
                    schema_str += "-" * 25 + "\n"
                    for fk in foreign_keys:
                        src_table_idx = column_names[fk[0]][0]
                        src_table = table_names[src_table_idx] if src_table_idx < len(table_names) else "unknown"
                        src_col = column_names[fk[0]][1]
                        
                        ref_table_idx = column_names[fk[1]][0]
                        ref_table = table_names[ref_table_idx] if ref_table_idx < len(table_names) else "unknown"
                        ref_col = column_names[fk[1]][1]
                        
                        schema_str += f"  {src_table}.{src_col} -> {ref_table}.{ref_col}\n"
                    schema_str += "\n"
                        
                return schema_str
                
        return f"Schema not found for database: {db_id}"
        
    def _create_dataset(self) -> Dataset:
        """Convert BIRD data to HuggingFace dataset format."""
        dataset_rows = []
        
        for item in self.bird_data:
            # Create the prompt with complete schema information
            schema = self._load_table_schema(item['db_id'])
            
            # Build comprehensive prompt with all columns
            prompt = f"{schema}\n\n"
            
            # Add evidence if available
            if item.get('evidence'):
                prompt += f"Evidence: {item['evidence']}\n\n"
            
            # Add the question
            prompt += f"Question: {item['question']}\n\n"
            prompt += "Write a SQL query to answer this question."
            
            # Store the gold SQL as answer
            answer = item.get('SQL', item.get('query', ''))
            
            # Create info dict for evaluation metadata
            info = {
                "db_id": item['db_id'],
                "difficulty": item.get('difficulty', 'moderate'),
                "evidence": item.get('evidence', ''),
                "task": "sql_generation",
                "question": item['question']
            }
            
            dataset_rows.append({
                "prompt": prompt,
                "answer": answer,
                "info": info
            })
            
        return Dataset.from_list(dataset_rows)
        
    def _get_default_system_prompt(self) -> str:
        """Return the default system prompt for SQL generation."""
        return """You are an expert SQL query writer. Given a database schema and a natural language question, generate the appropriate SQL query.

Format your response as follows:

<reasoning>
Step-by-step analysis of the question and required SQL components
</reasoning>

<sql>
Your SQL query here
</sql>

<explanation>
Brief explanation of what the query does
</explanation>

Guidelines:
1. Analyze the schema carefully
2. Use proper JOIN conditions when needed
3. Consider edge cases and NULL values
4. Ensure the query is syntactically correct
5. Optimize for readability and performance"""

    def _create_default_rubric(self, parser: XMLParser) -> Rubric:
        """Create the default rubric for SQL evaluation."""
        rubric = Rubric(parser=parser)
        
        # Main SQL evaluation function with new scoring system
        def sql_evaluation_reward_func(completion, answer, info, **kwargs) -> float:
            """
            Evaluate SQL query with new scoring system:
            - 0.0: Non-executable SQL queries
            - 0.1: Executable SQL queries
            - 1.0: Exact match and executable
            """
            sql = parser.parse_answer(completion)
            if not sql:
                return 0.0
                
            db_id = info.get('db_id', '')
            if not db_id:
                return 0.0
                
            db_path = self.databases_path / f"{db_id}/{db_id}.sqlite"
            if not db_path.exists():
                return 0.0
                
            with SQLExecutor(str(db_path)) as executor:
                # Execute generated SQL
                success, result, _ = executor.execute(sql)
                if not success:
                    return 0.0  # Non-executable SQL
                
                # SQL is executable, now check for exact match
                if not answer:
                    return 0.1  # Executable but no gold standard
                    
                # Execute gold SQL
                success_gold, result_gold, _ = executor.execute(answer)
                if not success_gold:
                    return 0.1  # Executable but gold standard failed
                    
                # Compare results for exact match
                if result == result_gold:
                    return 1.0  # Exact match and executable
                else:
                    return 0.1  # Executable but not exact match
                
        # Add the main evaluation function
        rubric.add_reward_func(sql_evaluation_reward_func, weight=1.0)
        
        return rubric
        
    def _normalize_sql(self, sql: str) -> str:
        """Normalize SQL query for comparison."""
        # Remove extra whitespace
        sql = ' '.join(sql.split())
        # Remove trailing semicolon
        sql = sql.rstrip(';')
        # Standardize quotes
        sql = sql.replace('"', "'")
        return sql
        
    def is_completed(self, messages: Messages, state: State, **kwargs) -> bool:
        """Check if the task is completed."""
        if state["turn"] >= self.max_turns:
            return True
            
        # Check if we have a valid SQL response
        if len(state["responses"]) > 0:
            last_response = messages[-1] if isinstance(messages, list) else messages
            sql = self.parser.parse_answer(last_response)
            
            if sql and not self.allow_refinement:
                return True
                
            # If execution was successful, we're done
            if sql and state.get("last_execution_success", False):
                return True
                
        return False
        
    def env_response(
        self, messages: Messages, state: State, **kwargs
    ) -> Tuple[Messages, State]:
        """Generate environment response with SQL execution feedback."""
        assert isinstance(messages, list)
        
        # Parse the SQL from the last message
        sql = self.parser.parse_answer(messages[-1]["content"])
        
        if not sql:
            return [{"role": "user", "content": "Please provide a valid SQL query in the <sql> tags."}], state
            
        # Execute the SQL if feedback is enabled
        if self.use_execution_feedback:
            # Get db_id from info dict
            info = state.get("info", {})
            db_id = info.get("db_id", "")
            db_path = self.databases_path / f"{db_id}/{db_id}.sqlite"
            
            if db_path.exists():
                with SQLExecutor(str(db_path)) as executor:
                    success, result, error = executor.execute(sql)
                    
                state["last_execution_success"] = success
                
                if success:
                    # Limit result size for feedback
                    if isinstance(result, list) and len(result) > 5:
                        result = result[:5] + [f"... and {len(result)-5} more rows"]
                    
                    feedback = f"SQL executed successfully.\nResult preview: {result}\n"
                    
                    # Check against gold standard if available
                    gold_sql = state.get("answer", "")
                    if gold_sql:
                        _, gold_result, _ = executor.execute(gold_sql)
                        if result == gold_result:
                            feedback += "✓ Results match expected output!"
                        else:
                            feedback += "⚠ Results differ from expected. Would you like to refine your query?"
                else:
                    feedback = f"SQL execution failed: {error}\nPlease fix the query."
            else:
                feedback = "Database not available for testing."
        else:
            feedback = "Query received. Please review for correctness."
            
        return [{"role": "user", "content": feedback}], state


def load_environment(
    bird_data_path: str = "/workspace/anushka/train/bird_train",
    num_train_examples: int = -1,
    num_eval_examples: int = 100,
    max_turns: int = 3,
    allow_refinement: bool = True,
    use_execution_feedback: bool = True,
    difficulty_filter: Optional[str] = None,  # 'simple', 'moderate', 'challenging'
    **kwargs
) -> SQLBirdEnv:
    """
    Load SQL BIRD environment for training.
    
    Args:
        bird_data_path: Path to BIRD dataset
        num_train_examples: Number of training examples (-1 for all)
        num_eval_examples: Number of evaluation examples
        max_turns: Maximum refinement turns
        allow_refinement: Allow query refinement based on feedback
        use_execution_feedback: Execute queries and provide feedback
        difficulty_filter: Filter by difficulty level
        **kwargs: Additional arguments for the environment
    
    Returns:
        SQLBirdEnv instance
    """
    env = SQLBirdEnv(
        bird_data_path=bird_data_path,
        max_turns=max_turns,
        allow_refinement=allow_refinement,
        use_execution_feedback=use_execution_feedback,
        **kwargs
    )
    
    # Apply difficulty filter if specified
    if difficulty_filter:
        dataset = env.dataset
        if dataset:
            dataset = dataset.filter(lambda x: x.get('difficulty', '') == difficulty_filter)
            env.dataset = dataset
    
    # Limit dataset size if specified
    if num_train_examples > 0 and env.dataset:
        env.dataset = env.dataset.select(range(min(num_train_examples, len(env.dataset))))
        
    # Create eval split
    if num_eval_examples > 0 and env.dataset and len(env.dataset) > num_eval_examples:
        total_size = len(env.dataset)
        train_size = total_size - num_eval_examples
        env.eval_dataset = env.dataset.select(range(train_size, total_size))
        env.dataset = env.dataset.select(range(train_size))
    
    return env 