
import os
import verifiers as vf

"""
MuSiQue XML GRPO Training - Simple Local

# install
vf-install vf-musique-xml -p ./verifiers/environments

# training
CUDA_VISIBLE_DEVICES=3 python verifiers/examples/grpo/train_musique_xml_simple.py
"""

# Load MuSiQue XML environment
vf_env = vf.load_environment(env_id="vf-musique-xml")

# Model configuration
model_name = os.environ.get("MODEL_PATH", "/workspace/anushka/models/Qwen3-4B")
# Removed torch_dtype from the call below
model, tokenizer = vf.get_model_and_tokenizer(model_name)
run_name = "musique_xml_" + model_name.split("/")[-1].lower()

# Training arguments
training_args = vf.grpo_defaults(run_name=run_name)
training_args.per_device_train_batch_size = 1
training_args.num_generations = 4
training_args.gradient_accumulation_steps = 4
training_args.max_tokens = 512
training_args.max_seq_len = 2048
training_args.max_steps = 100
training_args.mask_env_responses = True
training_args.max_grad_norm = 0.1
training_args.beta = 0.05
training_args.fp16 = True # Ensure FP16 is enabled here
training_args.gradient_checkpointing = True

# Evaluation and saving
training_args.eval_strategy = "steps"
training_args.eval_steps = 25
training_args.save_strategy = "steps"
training_args.save_steps = 50
training_args.dataloader_num_workers = 0
training_args.dataloader_pin_memory = False


# Print info
print(f"Training {model_name} on MuSiQue XML environment")
print(f"Dataset size: {len(vf_env.dataset)}")
print(f"Max steps: {training_args.max_steps}")

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
