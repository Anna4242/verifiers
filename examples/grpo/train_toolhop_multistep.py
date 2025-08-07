import verifiers as vf

"""
# install
vf-install vf-toolhop-multistep -p ./verifiers/environments

# quick eval
vf-eval vf-toolhop-multistep

# inference
CUDA_VISIBLE_DEVICES=0 vf-vllm --model $MODEL_PATH \
    --enforce-eager --disable-log-requests

# training
CUDA_VISIBLE_DEVICES=7 accelerate launch --num-processes 1 \
    --config-file verifiers/configs/zero3.yaml examples/grpo/train_toolhop_multistep.py
"""

# Model configuration - use environment variable or default
import os
model_name = os.environ.get("MODEL_PATH", "microsoft/DialoGPT-medium")
model, tokenizer = vf.get_model_and_tokenizer(model_name)

# Load ToolHop multistep environment
vf_env = vf.load_environment(env_id="vf-toolhop-multistep")

# Training arguments
run_name = "toolhop_multistep_" + model_name.split("/")[-1].lower()
args = vf.grpo_defaults(run_name=run_name)
args.per_device_train_batch_size = 2
args.num_generations = 4
args.gradient_accumulation_steps = 4
args.max_tokens = 1024
args.max_seq_len = 2048
args.max_steps = 500
args.eval_strategy = "steps"
args.eval_steps = 25
args.save_strategy = "steps" 
args.save_steps = 100
args.dataloader_num_workers = 0
args.dataloader_pin_memory = False
args.gradient_checkpointing = True
args.max_grad_norm = 0.1
args.beta = 0.05

# Print training info
print("=" * 60)
print("TOOLHOP MULTISTEP TRAINING CONFIGURATION")
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

# Create trainer
trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    peft_config=vf.lora_defaults(),
    args=args,
)

# Start training
trainer.train() 