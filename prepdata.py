import json
import random
import os
from datasets import load_dataset

def normalize_dolly(example):
    return {
        "instruction": example.get("instruction", "").strip(),
        "input": "",
        "output": example.get("response", "").strip()
    }

def normalize_finqa(example, max_chars=2000):
    pre = " ".join(example.get("pre_text", []) or [])
    table_rows = example.get("table", []) or []
    table_text = "\n".join(["\t".join(row) for row in table_rows]) if table_rows else ""
    post = " ".join(example.get("post_text", []) or [])
    answer = (example.get("final_result") or example.get("answer") or "").strip()
    if not answer:
        return None

    parts = []
    if pre:
        parts.append(pre)
    if table_text:
        parts.append("TABLE:\n" + table_text)
    if post:
        parts.append(post)
    context = "\n\n".join(parts).strip()

    if len(context) > max_chars:
        context = context[:max_chars].rsplit("\n", 1)[0] + "\n...[truncated]"

    question = example.get("question", "").strip()
    return {
        "instruction": question,
        "input": context,
        "output": answer
    }

def main():
    random.seed(42)
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)

    # load dolly
    print("loading dolly15k...")
    ds_dolly = load_dataset("databricks/databricks-dolly-15k")
    dolly_clean = [normalize_dolly(x) for x in ds_dolly["train"] if x.get("response")]
    print(f"loaded {len(dolly_clean)} dolly examples")

    # load finqa (parquet version)
    print("loading finqa...")
    ds_finqa = load_dataset("PTPReasoning/finqa")
    finqa_clean = []
    for split in ds_finqa.keys():
        for ex in ds_finqa[split]:
            norm = normalize_finqa(ex)
            if norm:
                finqa_clean.append(norm)
    print(f"loaded {len(finqa_clean)} finqa examples")

    # merge, dedupe, shuffle
    seen = set()
    merged = []
    def add(item):
        key = (item["instruction"].strip() + "||" + item["input"].strip())[:2000]
        if key not in seen and item.get("output"):
            seen.add(key)
            merged.append(item)

    for item in dolly_clean:
        add(item)
    for item in finqa_clean[:4000]:  # cap size for memory
        add(item)

    print(f"merged dataset size: {len(merged)} examples")
    random.shuffle(merged)

    split_idx = int(0.95 * len(merged))
    train, val = merged[:split_idx], merged[split_idx:]

    train_path = os.path.join(data_dir, "train.jsonl")
    val_path = os.path.join(data_dir, "val.jsonl")
    with open(train_path, "w", encoding="utf-8") as f:
        for item in train:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    with open(val_path, "w", encoding="utf-8") as f:
        for item in val:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"saved {len(train)} train and {len(val)} val examples")

if __name__ == "__main__":
    main()
