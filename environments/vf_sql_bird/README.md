# SQL BIRD Environment for Verifiers

A comprehensive environment for training models on SQL query generation using the BIRD (Big Bench for Large-scale Database Grounded Text-to-SQL Evaluation) dataset with GRPO (Group Relative Policy Optimization).

## Features

- **Multi-turn SQL Refinement**: Supports iterative query improvement based on execution feedback
- **Safe SQL Execution**: Uses temporary database copies for safe query testing
- **Comprehensive Evaluation**: Multiple reward functions including execution correctness, result similarity, and structural analysis
- **Flexible Configuration**: Customizable difficulty levels, dataset sizes, and training parameters
- **BIRD Dataset Integration**: Seamless integration with the BIRD benchmark dataset

## Installation

```bash
# Install the environment
vf-install sql_bird -p ./environments

# Or install in development mode
pip install -e ./environments/vf_sql_bird
```

## Dataset Setup

The environment expects the BIRD dataset in the following structure:

```
/workspace/anushka/train/bird_train/
├── train.json          # Training queries and expected SQL
├── train_databases/    # SQLite database files
│   └── [db_name]/
│       └── [db_name].sqlite
├── train_gold.sql      # Gold standard SQL queries
└── train_tables.json   # Table schemas and metadata
```

Download the BIRD dataset from the official repository and organize it according to this structure.

## Usage

### Basic Usage

```python
import verifiers as vf

# Load the environment
env = vf.load_environment(
    "vf-sql-bird",
    bird_data_path="/workspace/anushka/train/bird_train",
    num_train_examples=1000,
    max_turns=3,
    allow_refinement=True,
    use_execution_feedback=True
)

# Quick evaluation
results = env.evaluate(
    model="gpt-4-mini",
    num_examples=10,
    rollouts_per_example=3
)
```

### Training Configuration

```python
# For GRPO training
env = vf.load_environment(
    "vf-sql-bird",
    bird_data_path="/workspace/anushka/train/bird_train",
    num_train_examples=5000,
    num_eval_examples=500,
    max_turns=3,
    allow_refinement=True,
    use_execution_feedback=True,
    difficulty_filter="moderate"  # Optional: 'simple', 'moderate', 'challenging'
)
```

### Advanced Configuration

```python
from vf_sql_bird import SQLBirdEnv
from verifiers.parsers.xml_parser import XMLParser
from verifiers.rubrics.rubric import Rubric

# Custom parser
parser = XMLParser(
    fields=["reasoning", "sql", "explanation"],
    answer_field="sql"
)

# Custom system prompt
system_prompt = """You are an expert SQL analyst..."""

# Create environment with custom settings
env = SQLBirdEnv(
    bird_data_path="/path/to/bird/data",
    max_turns=5,
    allow_refinement=True,
    use_execution_feedback=True,
    system_prompt=system_prompt,
    parser=parser
)
```

## Environment Features

### SQL Execution Safety

The environment uses `SQLExecutor` class which:
- Creates temporary database copies for each query execution
- Implements query timeouts to prevent hanging
- Uses read-only mode to prevent data modification
- Provides detailed error messages for debugging

### Multi-turn Interaction

The environment supports iterative query refinement:
1. Model generates initial SQL query
2. Query is executed against the database
3. Feedback is provided (success/error, result preview)
4. Model can refine the query based on feedback
5. Process continues until completion or max turns reached

### Reward Functions

The default rubric includes multiple reward components:

1. **Execution Correctness** (weight: 1.0): Query executes without errors
2. **Result Similarity** (weight: 2.0): Results match gold standard
3. **Structure Similarity** (weight: 0.5): SQL structure matches expected patterns
4. **Format Compliance** (weight: 0.2): Proper XML formatting

### Response Format

The environment expects responses in XML format:

```xml
<reasoning>
Step-by-step analysis of the question and required SQL components
</reasoning>

<sql>
SELECT column_name FROM table_name WHERE condition;
</sql>

<explanation>
Brief explanation of what the query does
</explanation>
```

## Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `bird_data_path` | str | "/workspace/anushka/train/bird_train" | Path to BIRD dataset |
| `num_train_examples` | int | -1 | Number of training examples (-1 for all) |
| `num_eval_examples` | int | 100 | Number of evaluation examples |
| `max_turns` | int | 3 | Maximum refinement turns |
| `allow_refinement` | bool | True | Allow query refinement |
| `use_execution_feedback` | bool | True | Execute queries and provide feedback |
| `difficulty_filter` | str | None | Filter by difficulty ('simple', 'moderate', 'challenging') |

## Training Example

```bash
# Install environment
vf-install sql_bird -p ./environments

# Quick evaluation
vf-eval vf-sql-bird -m gpt-4-mini -n 10 -r 3

# Training with GRPO
accelerate launch --config-file configs/zero3.yaml \
    examples/grpo/train_sql_bird.py \
    --bird-data-path /workspace/anushka/train/bird_train \
    --num-train-examples 5000 \
    --max-turns 3 \
    --batch-size 16
```

## Troubleshooting

### Common Issues

**Database not found errors:**
- Ensure BIRD dataset is properly downloaded and organized
- Check that database files are in the correct directory structure
- Verify file permissions for database access

**SQL execution timeouts:**
- Increase timeout in SQLExecutor configuration
- Check for complex queries that might need optimization
- Verify database integrity

**Memory issues with large datasets:**
- Reduce `num_train_examples` parameter
- Use difficulty filtering to focus on specific complexity levels
- Consider batch processing for very large datasets

### Performance Tips

1. **Dataset Filtering**: Use `difficulty_filter` to focus on specific complexity levels
2. **Size Limiting**: Set appropriate `num_train_examples` for your hardware
3. **Turn Limiting**: Adjust `max_turns` based on your refinement needs
4. **Feedback Control**: Disable `use_execution_feedback` for faster training if execution feedback isn't needed

## Contributing

When contributing to this environment:

1. Ensure all SQL queries are properly sanitized
2. Add appropriate error handling for database operations
3. Test with various database schemas and query types
4. Follow the established XML parsing format
5. Update documentation for any new features

## License

This environment is part of the Verifiers framework and follows the same licensing terms.

## Citation

If you use this environment in your research, please cite:

```bibtex
@article{li2024bird,
  title={BIRD: A Big Bench for Large-scale Database Grounded Text-to-SQL Evaluation},
  author={Li, Jinyang and others},
  journal={arXiv preprint arXiv:2305.03111},
  year={2024}
}
``` 