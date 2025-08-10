
import verifiers as vf

"""
# Start vLLM server first:
CUDA_VISIBLE_DEVICES=1 vllm serve arcee-ai/AFM-4.5B \
  --host 0.0.0.0 --port 8000 \
  --served-model-name afm-4.5b \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 8192 \
  --trust-remote-code

# Then run training:
CUDA_VISIBLE_DEVICES=0 accelerate launch --num-processes 1 \
    --config-file configs/zero3.yaml examples/grpo/train_afm_bird.py
"""

# Load SQL BIRD environment
vf_env = vf.load_environment(env_id="vf-sql-bird", num_examples=1000)

# Model configuration - using AFM
model_name = "arcee-ai/AFM-4.5B"  # or "/workspace/anushka/model"
model_kwargs = {"torch_dtype": "bfloat16"}
model, tokenizer = vf.get_model_and_tokenizer(
    model_name,
    model_kwargs=model_kwargs
)
run_name = "afm_bird_grpo"

# Training arguments optimized for SQL generation
training_args = vf.grpo_defaults(run_name=run_name)
training_args.per_device_train_batch_size = 1
training_args.num_generations = 4
training_args.gradient_accumulation_steps = 8
training_args.max_tokens = 1024
training_args.max_seq_len = 2048
training_args.max_steps = 500
training_args.mask_env_responses = True
training_args.max_grad_norm = 0.1
training_args.beta = 0.01

# AFM recommended settings
training_args.temperature = 0.5
training_args.top_k = 50
training_args.top_p = 0.95

# Point to your vLLM server
training_args.vllm_server_host = "0.0.0.0"
training_args.vllm_server_port = 8000

# Evaluation and saving
training_args.eval_strategy = "steps"
training_args.eval_steps = 50
training_args.save_strategy = "steps" 
training_args.save_steps = 100
training_args.dataloader_num_workers = 0
training_args.dataloader_pin_memory = False
training_args.gradient_checkpointing = True

# Create GRPO trainer
trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    args=training_args,
    ref_model_cpu_offload=True,
    lora_config=vf.lora_defaults()
)

# Start training
trainer.train()
