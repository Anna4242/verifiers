import verifiers as vf

"""
# install
vf-install vf-llm-tool-exec -p ./verifiers/environments

# quick eval
vf-eval vf-llm-tool-exec -a '{"dataset_path":"toolhop_verifiers_format"}'

# inference (agent server)
CUDA_VISIBLE_DEVICES=0 vf-vllm --model $MODEL_PATH \
    --enforce-eager --disable-log-requests

# training (single GPU)
CUDA_VISIBLE_DEVICES=7 accelerate launch --num-processes 1 \
    --config-file verifiers/configs/zero3.yaml examples/grpo/train_llm_tool_exec.py
"""

# Model configuration - use environment variable or default
import os

model_name = os.environ.get("MODEL_PATH", "microsoft/DialoGPT-medium")
model, tokenizer = vf.get_model_and_tokenizer(
    model_name,
    use_liger=False,
    model_kwargs={"attn_implementation": "sdpa", "torch_dtype": "float16"},
)

# Load exec ToolEnv (ToolHop dataset with LLM tool execution via OpenRouter)
vf_env = vf.load_environment(
    env_id="vf-llm-tool-exec",
    dataset_path="toolhop_verifiers_format",
    exec_base_url="https://openrouter.ai/api/v1",
    exec_api_key_var="OPENROUTER_KEY",
    exec_model="moonshotai/kimi-k2",
)

# Training arguments
run_name = "llm_tool_exec_" + model_name.split("/")[-1].lower()
args = vf.grpo_defaults(run_name=run_name)
args.per_device_train_batch_size = 1
args.num_generations = 6
args.gradient_accumulation_steps = 12
args.max_tokens = 1536
args.max_seq_len = 2048
args.max_steps = 300
args.eval_strategy = "steps"
args.eval_steps = 25
args.save_strategy = "steps"
args.save_steps = 100
args.dataloader_num_workers = 0
args.dataloader_pin_memory = False
args.gradient_checkpointing = True
args.max_grad_norm = 0.1
args.beta = 0.1
# Enable W&B if desired (requires WANDB_API_KEY env var)
# args.report_to = "wandb"

print("=" * 60)
print("LLM TOOL EXEC TRAINING CONFIGURATION")
print("=" * 60)
print(f"Environment: {vf_env.__class__.__name__}")
print(f"Dataset size: {len(vf_env.dataset)}")
print(f"Number of reward functions: {len(vf_env.rubric.reward_funcs)}")
print("Reward functions:")
for i, func in enumerate(vf_env.rubric.reward_funcs, 1):
    print(f"  {i}. {func.__name__}")
print(f"Model: {model_name}")
print(f"Run name: {run_name}")
print(f"Max steps: {args.max_steps}")
print(f"Batch size per device: {args.per_device_train_batch_size}")
print("=" * 60)

trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    peft_config=vf.lora_defaults(),
    args=args,
)

trainer.train()


