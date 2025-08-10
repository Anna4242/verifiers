
import os
# Talk to vLLM over HTTP only (avoid PyNccl issues)
os.environ.setdefault("VLLM_USE_PYNCCL","0")
os.environ.setdefault("VLLM_USE_NCCL_COMM","0")

import verifiers as vf
from transformers import get_linear_schedule_with_warmup
from bitsandbytes.optim import Adam8bit
from peft import LoraConfig, get_peft_model, TaskType

# --- Environment ---
env = vf.load_environment(
    env_id="vf-sql-bird-new",
    bird_data_path="/workspace/anushka/train/bird_train",
    num_train_examples=2000,
    num_eval_examples=200,
)

# --- Base model ---
model_name = "/workspace/anushka/models/Qwen3-4B"
model, tok = vf.get_model_and_tokenizer(model_name, model_kwargs={"torch_dtype":"bfloat16"})
# Gradient checkpointing works best with use_cache disabled
if hasattr(model, "config"):
    try:
        model.config.use_cache = False
    except Exception:
        pass

# --- LoRA (keep trainable params tiny) ---
lora_cfg = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
)
model = get_peft_model(model, lora_cfg)

# Log trainable vs total params
tot = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"[LoRA] trainable params: {trainable/1e6:.2f}M / total {tot/1e6:.2f}M "
      f"({100*trainable/tot:.3f}%)")

# --- Training args ---
args = vf.grpo_defaults(run_name="qwen3_4b_sql_bird_new_grpo_lora8bit")
args.per_device_train_batch_size = 1
args.gradient_accumulation_steps = 1
args.gradient_checkpointing = True
args.bf16 = True

# Keep loads modest & consistent
args.num_generations = 1        # <- avoids reward shape mismatch
args.max_tokens = 512
args.max_seq_len = 2048
args.max_steps = 500

args.mask_env_responses = True
args.max_grad_norm = 0.2
args.beta = 0.02
args.logging_steps = 10
args.eval_strategy = "steps"; args.eval_steps = 100
args.save_strategy = "steps"; args.save_steps = 200
args.dataloader_num_workers = 0
args.dataloader_pin_memory = False
# Drop incomplete last batch to avoid grouping issues
args.dataloader_drop_last = True

# vLLM server info (must match your running engine)
args.vllm_server_host = "127.0.0.1"
args.vllm_server_port = 8000
args.vllm_server_model = "Qwen3-4B"

# --- Optimizer & scheduler (8-bit Adam on LoRA params only) ---
trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = Adam8bit(trainable_params, lr=1e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)

num_update_steps = args.max_steps
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=max(1, int(0.03 * num_update_steps)),
    num_training_steps=num_update_steps,
)

# --- Trainer ---
trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tok,
    env=env,
    args=args,
    ref_model_cpu_offload=True,
    optimizers=(optimizer, scheduler),   # override default torch Adam
)

trainer.train()
