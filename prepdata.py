import json
from datasets import load_dataset
import random

def normalize_dolly(example):
    return {
        "instruction": example.get("instruction", "").strip(),
        "input": "",
        "output": example.get("response", "").strip()
    }


def main():
    # Load dataset
    ds_dolly = load_dataset("databricks/databricks-dolly-15k")
    print("Dolly splits:", ds_dolly.keys())
    # Filter and normalize
    dolly_clean = [normalize_dolly(x) for x in ds_dolly['train'] if x.get("response")]
    random.shuffle(dolly_clean)
    split_idx = int(0.95 * len(dolly_clean))
    train = dolly_clean[:split_idx]
    val = dolly_clean[split_idx:]
    # Save to JSONL

    with open("data/train.jsonl", "w", encoding="utf-8") as f:
        for item in train:
            f.write(json.dumps(item) + "\n")
    with open("data/val.jsonl", "w", encoding="utf-8") as f:
        for item in val:
            f.write(json.dumps(item) + "\n")

    print(f"Saved {len(train)} train and {len(val)} val examples.")


if __name__ == "__main__":
    main()
