"""
GT-Quant Qwen3-8B fine-tuning v0.2 (iteration 2: more consolidation data, 4 epochs).

Same recipe as train_llm_qwen3.py (QLoRA r=64, alpha=16, lr 2e-4,
bs 2x4, seq 2048, 4-bit, non-thinking qwen3-instruct template) but:
  - v0.3 dataset regenerated with +400 relaxed-threshold consolidation examples
    (3720 train rows vs 3320)
  - 4 epochs instead of 3 (v0.1 eval loss was still decreasing at epoch 3)

Output: models/gtquant-8b-qwen3-v0.2/

Run on the RTX 4080 (Ollama must be STOPPED first — see nvidia-smi):
    source venv/bin/activate
    python train_llm_qwen3_v2.py
"""
import torch
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# ─── CONFIG ───
MODEL_NAME = "unsloth/Qwen3-8B"
MAX_SEQ_LENGTH = 2048
TRAIN_PATH = "data/llm_v03_train.jsonl"
VAL_PATH = "data/llm_v03_val.jsonl"
OUTPUT_DIR = "models/gtquant-8b-qwen3-v0.2"
LORA_R = 64
LORA_ALPHA = 16
LEARNING_RATE = 2e-4
NUM_EPOCHS = 4
BATCH_SIZE = 2
GRAD_ACCUM = 4

SYSTEM = ("You are GT-Quant, a crypto trading analyst. Analyze the multi-timeframe "
          "context and output a JSON decision. Consider: 1m for execution timing, 5m "
          "for primary signals, 15m/30m for trend confirmation, 1h/4h for regime.")

# ─── LOAD MODEL ───
print("[*] Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
)

# Non-thinking Qwen3 template: assistant turns have no <think> block.
tokenizer = get_chat_template(tokenizer, "qwen3-instruct")

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

# ─── DATASET ───
print("[*] Loading dataset...")


def format_prompt(example):
    # Plain ChatML without a <think> block — matches the qwen3-instruct
    # non-thinking template so inference emits JSON directly.
    return f"""<|im_start|>system
{SYSTEM}<|im_end|>
<|im_start|>user
{example['input']}<|im_end|>
<|im_start|>assistant
{example['output']}<|im_end|>"""


train_ds = load_dataset("json", data_files=TRAIN_PATH, split="train")
val_ds = load_dataset("json", data_files=VAL_PATH, split="train")
train_ds = train_ds.map(lambda x: {"text": format_prompt(x)})
val_ds = val_ds.map(lambda x: {"text": format_prompt(x)})
print(f"[*] train {len(train_ds)} | val {len(val_ds)}")

# ─── TRAIN ───
print("[*] Starting Qwen3-8B v0.1 training...")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        warmup_steps=5,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        eval_strategy="epoch",
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        output_dir=OUTPUT_DIR,
        save_strategy="epoch",
        report_to="none",
    ),
)

trainer_stats = trainer.train()
print(f"[*] Training complete. Final loss: {trainer_stats.training_loss:.4f}")

# ─── SAVE ───
print("[*] Saving LoRA adapter...")
model.save_pretrained(f"{OUTPUT_DIR}/lora_adapter")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/lora_adapter")

print("[*] Merging and saving 16-bit (for vLLM)...")
model.save_pretrained_merged(f"{OUTPUT_DIR}/merged_16bit", tokenizer, save_method="merged_16bit")

print("[*] Exporting GGUF Q4_K_M (for Ollama)...")
model.save_pretrained_gguf(f"{OUTPUT_DIR}/gguf", tokenizer, quantization_method="q4_k_m")

print(f"[*] Done. Artifacts in {OUTPUT_DIR}/")
print("  - lora_adapter/    : LoRA weights")
print("  - merged_16bit/    : full model for vLLM")
print("  - gguf/            : GGUF for Ollama")
