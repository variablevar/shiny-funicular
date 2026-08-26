"""
GT-Quant LLM Fine-Tuning with Unsloth
Run on your RTX 4080 (16GB)
"""
import torch
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# ─── CONFIG ───
MODEL_NAME = "unsloth/Qwen2.5-7B-Instruct"
MAX_SEQ_LENGTH = 2048
DATASET_PATH = "data/gtquant_llm_dataset.jsonl"
OUTPUT_DIR = "models/gtquant-7b-v1"
LORA_R = 64
LORA_ALPHA = 16
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
BATCH_SIZE = 2
GRAD_ACCUM = 4

# ─── LOAD MODEL ───
print("[*] Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,  # Auto-detect
    load_in_4bit=True,
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=LORA_ALPHA,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

# ─── PREPARE DATASET ───
print("[*] Loading dataset...")

def format_prompt(example):
    """Convert JSONL to chat format."""
    return f"""<|im_start|>system
You are GT-Quant, a crypto perpetual futures trading analyst. Analyze market data and output valid JSON with keys: regime, bias, confidence, risk, reasoning.<|im_end|>
<|im_start|>user
{example['input']}<|im_end|>
<|im_start|>assistant
{example['output']}<|im_end|>"""

dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
dataset = dataset.map(lambda x: {"text": format_prompt(x)})

# ─── TRAIN ───
print("[*] Starting training...")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        warmup_steps=5,
        max_steps=-1,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        output_dir=OUTPUT_DIR,
        save_strategy="epoch",
    ),
)

trainer_stats = trainer.train()

print(f"[*] Training complete. Final loss: {trainer_stats.training_loss:.4f}")

# ─── SAVE ───
print("[*] Saving model...")

# Save LoRA adapters
model.save_pretrained(f"{OUTPUT_DIR}/lora_adapter")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/lora_adapter")

# Save merged 16-bit (optional, for vLLM)
print("[*] Merging and saving 16-bit...")
model.save_pretrained_merged(f"{OUTPUT_DIR}/merged_16bit", tokenizer, save_method="merged_16bit")

# Export to GGUF for Ollama
print("[*] Exporting to GGUF (Q4_K_M)...")
model.save_pretrained_gguf(
    f"{OUTPUT_DIR}/gguf",
    tokenizer,
    quantization_method="q4_k_m",
)

print(f"[*] Done. Files saved to {OUTPUT_DIR}/")
print("  - lora_adapter/    : LoRA weights for further training")
print("  - merged_16bit/    : Full model for vLLM")
print("  - gguf/            : GGUF for Ollama")