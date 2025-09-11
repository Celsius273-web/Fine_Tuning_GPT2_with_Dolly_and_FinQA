import os
import json
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
import torch


def create_sample_data():
    """Create sample training data if files don't exist"""
    os.makedirs("data", exist_ok=True)

    if not os.path.exists("data/train.jsonl"):
        sample_train_data = [
            {"instruction": "Write a short story", "input": "about a cat",
             "output": "Once upon a time, there was a curious cat named Whiskers who loved to explore the garden."},
            {"instruction": "Explain a concept", "input": "what is photosynthesis",
             "output": "Photosynthesis is the process by which plants convert sunlight into energy using chlorophyll."},
            {"instruction": "Answer a question", "input": "what is the capital of France",
             "output": "The capital of France is Paris."},
            {"instruction": "Complete the sentence", "input": "The weather today is",
             "output": "The weather today is sunny and pleasant with a gentle breeze."},
            {"instruction": "Write a poem", "input": "about the ocean",
             "output": "Blue waves dance beneath the sky, where seagulls soar and dolphins fly."}
        ]

        with open("data/train.jsonl", "w") as f:
            for item in sample_train_data:
                f.write(json.dumps(item) + "\n")
        print("Created sample training data at data/train.jsonl")

    if not os.path.exists("data/val.jsonl"):
        sample_val_data = [
            {"instruction": "Write a haiku", "input": "about winter",
             "output": "Snow falls gently down, covering the earth in white, peaceful winter scene."},
            {"instruction": "Explain", "input": "how rain forms",
             "output": "Rain forms when water vapor in clouds condenses into droplets that become heavy enough to fall."}
        ]

        with open("data/val.jsonl", "w") as f:
            for item in sample_val_data:
                f.write(json.dumps(item) + "\n")
        print("Created sample validation data at data/val.jsonl")


def main():
    # Create sample data if needed
    create_sample_data()

    # Check if data files exist
    if not os.path.exists("data/train.jsonl") or not os.path.exists("data/val.jsonl"):
        print("Error: Data files not found. Please ensure data/train.jsonl and data/val.jsonl exist.")
        return

    # Use standard GPT-2 model (much smaller and faster)
    model_name = "gpt2/gpt2_model"

    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Set padding token to EOS token (as mentioned in the article)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model {model_name}...")
    # Use CPU and float32 for Mac compatibility
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
    # Save both model and tokenizer
    output_dir = "model/gpt2_lora_finetuned"
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Fine-tuning complete! Model saved to {output_dir}")
    print("\nTo test the model, you can use: python inference.py")


if __name__ == "__main__":
    main()