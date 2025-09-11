import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
import os

# Check if data files exist
if not os.path.exists("data/train.jsonl") or not os.path.exists("data/val.jsonl"):
    print("Error: Data files not found. Please ensure data/train.jsonl and data/val.jsonl exist.")
    exit(1)

# Use GPT-2 for testing (much smaller, ~500MB vs 8GB)
model_name = "gpt2/gpt2_model"

print(f"Loading tokenizer for {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

print(f"Loading model {model_name}...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    low_cpu_mem_usage=True,
    dtype=torch.float16
)

model.gradient_checkpointing_enable()

print("Loading dataset...")
dataset = load_dataset("json", data_files={
    "train": "data/train.jsonl",
    "validation": "data/val.jsonl"
})

def tokenize_fn(batch):
    texts = [
        f"Instruction: {instr}\nInput: {inp}\nOutput: {out}"
        for instr, inp, out in zip(batch["instruction"], batch["input"], batch["output"])
    ]
    return tokenizer(texts, truncation=True, padding="max_length", max_length=512)

print("Tokenizing dataset...")
tokenized_datasets = dataset.map(tokenize_fn, batched=True)

# LoRA configuration for GPT-2
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["c_attn", "c_proj"],  # GPT-2 specific modules
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

print("Applying LoRA...")
model = get_peft_model(model, lora_config)

# Print trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
all_params = sum(p.numel() for p in model.parameters())
print(f"Trainable parameters: {trainable_params:,}")
print(f"All parameters: {all_params:,}")
print(f"Percentage of trainable parameters: {100 * trainable_params / all_params:.2f}%")

training_args = TrainingArguments(
    output_dir="./results",
    #evaluation_strategy="epoch",
    logging_dir="./logs",
    logging_steps=5,
    per_device_train_batch_size=2,  # Can use larger batch size with GPT-2
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    learning_rate=2e-4,
    #fp16=True,
    save_strategy="epoch",
    save_total_limit=1,
    report_to="none",
    warmup_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"].select(range(min(50, len(tokenized_datasets["train"])))),
    eval_dataset=tokenized_datasets["validation"].select(range(min(10, len(tokenized_datasets["validation"])))),
    tokenizer=tokenizer
)

print("🚀 Starting training...")
trainer.train()

print("💾 Saving model...")
model.save_pretrained("./gpt2_lora")
tokenizer.save_pretrained("./gpt2_lora")

print("✅ Fine-tuning complete! Model saved to ./gpt2_lora")