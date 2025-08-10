cat > vf_sql_bird_new.py << 'EOF'
"""
SQL BIRD Environment for Verifiers Framework
=============================================

This environment enables training models on SQL generation using the BIRD dataset
with GRPO (Group Relative Policy Optimization).

Dataset Structure:
- Creates Hugging Face Dataset with:
  * 'prompt': Chat-format prompt (list of messages): [system, user]
  * 'answer': Gold standard SQL query (string)
  * 'info': Evaluation metadata (dict) containing db_id, difficulty, etc.

Scoring System:
- 0.0: Non-executable SQL queries
- 0.1: Executable SQL queries
- 1.0: Exact match and executable queries

Directory structure expected:
/workspace/anushka/train/bird_train/
├── train.json
├── train_databases/
├── train_gold.sql
└── train_tables.json

Installation:
1. Save this file as: environments/vf_sql_bird_new/vf_sql_bird_new.py
2. Create pyproject.toml in the same directory
3. Install: vf-install vf_sql_bird_new -p ./environments
"""

import json
import os
import sqlite3
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import tempfile
import shutil

from datasets import Dataset

import verifiers as vf
from verifiers.envs.multiturn_env import MultiTurnEnv
from verifiers.types import Messages, State
from verifiers.parsers.xml_parser import XMLParser
from verifiers.rubrics.rubric import Rubric


class SQLExecutor:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.temp_db_path = None

    def __enter__(self):
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.temp_db_path = self.temp_db.name
        self.temp_db.close()
        shutil.copy2(self.db_path, self.temp_db_path)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.temp_db_path and os.path.exists(self.temp_db_path):
            os.unlink(self.temp_db_path)

    def execute(self, query: str, timeout: float = 5.0) -> Tuple[bool, Any, str]:
        if not self.temp_db_path:
            return False, None, "Database not initialized"
        try:
            conn = sqlite3.connect(self.temp_db_path, timeout=timeout)
            conn.execute("PRAGMA query_only = ON")
            cursor = conn.cursor()
            cursor.execute(query)
            ql = query.lower().strip()
            if ql.startswith('select'):
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
    - 'prompt': Chat-format [system, user]
    - 'answer': Gold SQL
    - 'info': metadata

    Scoring:
    - 0.0 non-executable
    - 0.1 executable
    - 1.0 exact match and executable
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

        # Resolve system prompt string first (so we can embed it in prompts)
        system_prompt_str = system_prompt if system_prompt is not None else self._get_default_system_prompt()

        # Load BIRD dataset
        self.bird_data = self._load_bird_data()

        # Build dataset with chat-format prompts [system, user]
        dataset = self._create_dataset(system_prompt=system_prompt_str)

        # Parser
        if parser is None:
            parser = XMLParser(fields=["reasoning", "sql", "explanation"], answer_field="sql")

        # Rubric
        if rubric is None:
            rubric = self._create_default_rubric(parser)

        # IMPORTANT: pass system_prompt=None because prompts are already chat-formatted with system message
        super().__init__(
            dataset=dataset,
            system_prompt=None,
            parser=parser,
            rubric=rubric,
            max_turns=max_turns,
            message_type="chat",
            **kwargs
        )

    def _load_bird_data(self) -> List[Dict]:
        train_file = self.bird_data_path / "train.json"
        with open(train_file, 'r') as f:
            data = json.load(f)
        return data

    def _load_table_schema(self, db_id: str) -> str:
        tables_file = self.bird_data_path / "train_tables.json"
        with open(tables_file, 'r') as f:
            db_infos = json.load(f)
        for db_info in db_infos:
            if db_info["db_id"] == db_id:
                schema_str = f"Database: {db_id}\n"
                schema_str += "=" * (len(db_id) + 10) + "\n\n"
                table_names = db_info.get("table_names", [])
                column_names = db_info.get("column_names", [])
                column_types = db_info.get("column_types", [])
                foreign_keys = db_info.get("foreign_keys", [])
                primary_keys = db_info.get("primary_keys", [])
                for table_idx, table_name in enumerate(table_names):
                    schema_str += f"Table: {table_name}\n"
                    schema_str += "-" * (len(table_name) + 7) + "\n"
                    rows = []
                    for col_idx, (t_idx, col_name) in enumerate(column_names):
                        if t_idx == table_idx:
                            col_type = column_types[col_idx] if col_idx < len(column_types) else "TEXT"
                            pk = " (PK)" if col_idx in primary_keys else ""
                            fk_info = ""
                            for fk in foreign_keys:
                                if fk[0] == col_idx:
                                    ref_t_idx = column_names[fk[1]][0]
                                    ref_table = table_names[ref_t_idx] if ref_t_idx < len(table_names) else "unknown"
                                    ref_col = column_names[fk[1]][1]
                                    fk_info = f" -> {ref_table}.{ref_col}"
                                    break
                            rows.append(f"  - {col_name} ({col_type}){pk}{fk_info}")
                    schema_str += "Columns:\n" + ("\n".join(rows) if rows else "No columns found")
                    schema_str += "\n\n"
                if foreign_keys:
                    schema_str += "Foreign Key Relationships:\n" + "-" * 25 + "\n"
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

    def _create_dataset(self, system_prompt: Optional[str] = None) -> Dataset:
        rows = []
        for item in self.bird_data:
            schema = self._load_table_schema(item["db_id"])
            user_prompt = f"{schema}\n\n"
            if item.get("evidence"):
                user_prompt += f"Evidence: {item['evidence']}\n\n"
            user_prompt += f"Question: {item['question']}\n\n"
            user_prompt += "Write a SQL query to answer this question."

            # chat-format prompt (list of messages)
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})

            answer = item.get("SQL", item.get("query", ""))
            info = {
                "db_id": item["db_id"],
                "difficulty": item.get("difficulty", "moderate"),
                "evidence": item.get("evidence", ""),
                "task": "sql_generation",
                "question": item["question"],
            }
            rows.append({"prompt": messages, "answer": answer, "info": info})
        return Dataset.from_list(rows)

    def _get_default_system_prompt(self) -> str:
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
        rubric = Rubric(parser=parser)

        def sql_eval(completion, answer, info, **kwargs) -> float:
            sql = parser.parse_answer(completion)
            if not sql:
                return 0.0
            db_id = info.get("db_id", "")
            if not db_id:
                return 0.0
            db_path = self.databases_path / f"{db_id}/{db_id}.sqlite"
            if not db_path.exists():
                return 0.0
            with SQLExecutor(str(db_path)) as executor:
                ok, result, _ = executor.execute(sql)
                if not ok:
                    return 0.0
                if not answer:
                    return 0.1
                ok_gold, result_gold, _ = executor.execute(answer)
                if not ok_gold:
                    return 0.1
                return 1.0 if result == result_gold else 0.1

        rubric.add_reward_func(sql_eval, weight=1.0)
        return rubric

    def is_completed(self, messages: Messages, state: State, **kwargs) -> bool:
        if state["turn"] >= self.max_turns:
            return True
        if len(state["responses"]) > 0:
            # Pass the full conversation to the parser (expects list of messages)
            sql = self.parser.parse_answer(messages)
            if sql and not self.allow_refinement:
                return True
            if sql and state.get("last_execution_success", False):
                return True
        return False

    def env_response(self, messages: Messages, state: State, **kwargs) -> Tuple[Messages, State]:
        assert isinstance(messages, list)
        # Parse SQL from the full conversation (list of messages)
        sql = self.parser.parse_answer(messages)
        if not sql:
            return [{"role": "user", "content": "Please provide a valid SQL query in the <sql> tags."}], state

        if self.use_execution_feedback:
            info = state.get("info", {})
            db_id = info.get("db_id", "")
            db_path = self.databases_path / f"{db_id}/{db_id}.sqlite"
            if db_path.exists():
                with SQLExecutor(str(db_path)) as executor:
                    success, result, error = executor.execute(sql)
                state["last_execution_success"] = success
                if success:
                    if isinstance(result, list) and len(result) > 5:
                        result = result[:5] + [f"... and {len(result)-5} more rows"]
                    feedback = f"SQL executed successfully.\nResult preview: {result}\n"
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
    difficulty_filter: Optional[str] = None,
    **kwargs
) -> SQLBirdEnv:
    env = SQLBirdEnv(
        bird_data_path=bird_data_path,
        max_turns=max_turns,
        allow_refinement=allow_refinement,
        use_execution_feedback=use_execution_feedback,
        **kwargs
    )
    if difficulty_filter:
        dataset = env.dataset
        if dataset:
            dataset = dataset.filter(lambda x: x.get('difficulty', '') == difficulty_filter)
            env.dataset = dataset
    if num_train_examples > 0 and env.dataset:
        env.dataset = env.dataset.select(range(min(num_train_examples, len(env.dataset))))
    if num_eval_examples > 0 and env.dataset and len(env.dataset) > num_eval_examples:
        total_size = len(env.dataset)
        train_size = total_size - num_eval_examples
        env.eval_dataset = env.dataset.select(range(train_size, total_size))
        env.dataset = env.dataset.select(range(train_size))
    return env
EOF