# Fine-Tuning GPT-2 with LoRA on Mixed Instruction and Financial Datasets

This project demonstrates how to fine-tune a base language model (GPT-2) using Low-Rank Adaptation (LoRA) on a single local machine. By blending general instruction-following data with domain-specific financial reasoning data, the goal was to learn the practical engineering constraints of parameter-efficient fine-tuning (PEFT) and observe how alignment affects a smaller base model's response structures.

## Core Goals

* Implement parameter-efficient fine-tuning (PEFT) using Hugging Face and LoRA.
* Work around local hardware and memory constraints by utilizing gradient accumulation.
* Evaluate the structural differences in text generation between a raw pre-trained base model and an adapter-aligned model.

## How the Pipeline Works

### 1. Data Ingestion & Formatting (`prepdata.py`)

The data pipeline loads and standardizes two distinct datasets into a unified instruction input output schema:

* **Databricks Dolly 15k:** Used for general, open-ended instruction following.
* **FinQA:** Financial dataset containing complex tables, surrounding text reports, and quantitative questions.

To prevent out-of-memory errors on a standard laptop, the script automatically truncates massive financial contexts and caps the maximum number of FinQA examples injected into the training pool. It shuffles the merged datasets and splits them into a 95% training set (`train.jsonl`) and a 5% validation set (`val.jsonl`).

### 2. Low-Resource Training Configuration (`tune.py`)

Training was executed entirely locally on a laptop, taking roughly 20 to 30 minutes for a full training pass. To make this possible without crashing local memory, the script uses several optimization techniques:

* **Float32 Precision:** Standardized to float32 execution for stability on laptop CPU/MPS environments.
* **LoRA Target Modules:** Targets the attention layer query and value projections (`c_attn`) in GPT-2, freezing the rest of the network to minimize trainable parameters.
* **Micro-Batching:** Combines a small batch size of 2 with gradient accumulation steps to simulate larger batch workloads without massive RAM spikes.

### 3. Inference & Comparison (`inference.py`)

To test performance, both the original raw base model and the newly trained LoRA adapter model are loaded into memory. They are given a uniform list of test prompts (`test.json`), and their raw text generations are saved side-by-side into `model_comparison.json`.

The test evaluation enables `do_sample=True` with a temperature of 0.7 for the fine-tuned model to allow for natural creativity, while evaluating the structural consistency against the base model.

## Dependencies and Setup

Make sure you have Python 3.11 or 3.12 installed. Clone the repository and install the specific library versions required:

```bash
git clone https://github.com/Celsius273-web/Fine_Tuning_GPT2_with_Dolly_and_FinQA
cd Fine_Tuning_GPT2_with_Dolly_and_FinQA
pip install -r requirements.txt

```

Ensure your `requirements.txt` file contains:

```text
datasets==4.0.0
peft==0.17.1
torch==2.8.0
transformers==4.56.1

```

## Running the Project

1. **Prepare the Datasets:** Download, normalize, and split the data into local JSONL files:
```bash
python prepdata.py

```


2. **Train the Adapter:** Fine-tune GPT-2 via LoRA locally. The adapter weights will save directly to `model/gpt2_lora_finetuned/`:
```bash
python tune.py

```


3. **Run Evaluation:** Generate responses from both models and compare the results:
```bash
python inference.py

```



## Key Findings & Results

The fine-tuned adapter shows a massive improvement in structural language capability compared to the raw base model.

* **Syntax Alignment:** Raw GPT-2 frequently falls into infinite repetition loops or simply echoes the input prompt back to the terminal. The LoRA adapter successfully aligns the model to recognize the `Instruction/Input/Output` pattern, generating distinct answers that attempt to resolve the prompt directly.
* **Financial Data Performance:** While the model’s small scale limits its deep reasoning capacity on complex multi-step math, it shows a noticeable improvement in recognizing and attempting to extract financial concepts from the structured data fields compared to the un-tuned base model.

A full breakdown of prompt outputs can be reviewed inside `model_comparison.json`.
