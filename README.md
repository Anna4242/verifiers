# Verifiers SQL Environment

This repository contains a SQL generation environment for the [Verifiers framework](https://github.com/arcee-ai/verifiers).



1. **Add the SQL environment files:**
   - Copy `environments/vf_sql/` to `verifiers/environments/vf_sql/`
   - Copy `examples/grpo/train_vf_sql.py` to `verifiers/examples/grpo/train_vf_sql.py`

3. **Install the environment:**
   ```bash
   vf-install vf_sql -p ./environments
   ```

## Usage

### Quick Evaluation
```bash
vf-eval vf-sql \
  --api-base-url "http://127.0.0.1:8000/v1" \
  --api-key-var EMPTY \
  --model "/path/to/your/model" \
  --num-examples 2 --rollouts-per-example 1 \
  --temperature 0.1 --max-tokens 2048
```

### Training
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num-processes 4 \
  --config-file configs/zero3.yaml examples/grpo/train_vf_sql.py --data-path /path/to/dataset
```

## Dataset Structure

Ensure your dataset follows this structure:
```
<DATA_PATH>/
├── train.json
├── train_databases/
│   └── <db_id>/<db_id>.sqlite
├── train_gold.sql
└── train_tables.json
```

This environment was tested with the [BIRD dataset](https://bird-bench.github.io/) as an example.



## Scoring

- **0.0**: Non-executable SQL queries
- **0.1**: Executable SQL queries
- **1.0**: Exact match and executable queries
