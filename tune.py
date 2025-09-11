import os
import json
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
import torch


def main():
    if not os.path.exists("data/train.jsonl") or not os.path.exists("data/val.jsonl"):
        print("Error: Data files not found make sure data/train.jsonl and data/val.jsonl exist.")
        return

    model_name = "gpt2/gpt2_model"

    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # Use float32 instead of float16 for CPU
        low_cpu_mem_usage=True
    )

    print("Loading dataset...")
    dataset = load_dataset("json", data_files={
        "train": "data/train.jsonl",
        "validation": "data/val.jsonl"
    })

    def tokenize_function(examples):
        # Format the text as instruction-input-output
        texts = []
        for i in range(len(examples["instruction"])):
            text = f"Instruction: {examples['instruction'][i]}\nInput: {examples['input'][i]}\nOutput: {examples['output'][i]}"
            texts.append(text)

        inputs = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=128  # Same as article
        )
        inputs["labels"] = inputs["input_ids"].copy()
        return inputs

    print("Tokenizing dataset...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # LoRA configuration optimized for GPT-2 and Mac
    lora_config = LoraConfig(
        r=8,  # Low rank
        lora_alpha=16,
        target_modules=["c_attn", "c_proj"],  # GPT-2 attention modules
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    print("Applying LoRA adaptation...")
    model = get_peft_model(model, lora_config)

    # Print trainable parameters info
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"All parameters: {all_params:,}")
    print(f"Percentage of trainable parameters: {100 * trainable_params / all_params:.2f}%")

    # Training arguments optimized for Mac
    training_args = TrainingArguments(
        output_dir="finetuneLLM/results",
        #evaluation_strategy='epoch',  # Evaluate each epoch like in article
        logging_dir="finetuneLLM/logs",
        per_device_train_batch_size=2,  # Small batch size for Mac
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=2, #Updates batches slower less work
        num_train_epochs=1,
        learning_rate=2e-4,
        save_strategy="epoch",
        save_total_limit=1,
        warmup_steps=10,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer
    )

    print("Starting training...")
    trainer.train()

    print("Saving fine-tuned model...")
    output_dir = "model/gpt2_lora_finetuned"
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Fine-tuning complete! Model saved to {output_dir}")
    print("\nTo test the model, you can use: python inference.py")


if __name__ == "__main__":
    main()