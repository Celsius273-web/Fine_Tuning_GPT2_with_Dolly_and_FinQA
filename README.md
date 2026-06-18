# GPT-2 LoRA Fine Tuning for Instruction Following and Financial QA

Tech Stack: Python, PyTorch, Hugging Face Transformers, PEFT, LoRA, GPT-2

Fine-tuned GPT-2 with Low-Rank Adaptation (LoRA) on a mixture of instruction-following (Dolly-15k) and financial reasoning (FinQA) datasets. GPT-2 was intentionally selected to balance model capability with the compute and memory limitations of my laptop hardware.

### Skills Demonstrated

* **LLM FineTuning & Alignment**
* **Parameter Efficient Fine Tuning (PEFT) / LoRA**
* **Hugging Face Transformers ecosystem**
* **Dataset Engineering & Multi Task Blending**
* **Qualitative Model Evaluation & Benchmarking**

---

## Measured Results & Evaluation

A manual evaluation of **30 test prompts** was conducted to compare the base pre-trained model directly against the LoRA-aligned model.

### Findings

* **Repetition Loop Recovery:** The base GPT-2 model entered repetitive generation patterns on 24/30 prompts. The LoRA model produced non-repetitive outputs on all 24 affected prompts.


* **Instruction Format Adherence:** The fine-tuned model correctly adopted the structured `Instruction/Input/Output` framing or responded with direct summary outputs rather than echoing the prompt question.


* **Domain Awareness:** While limited by its small parameters, the fine-tuned adapter showed an enhanced ability to recognize financial syntax (e.g., balance sheets, ROE concepts).



### Before vs. After Comparison (`model_comparison.json`)

| Prompt | Base GPT-2 Model (Raw) | Fine-Tuned Model (GPT-2 + LoRA) |
| --- | --- | --- |
| **"Tell me a story about a robot."**<br> | *"I'm a robot. I'm a robot. I'm a robot. I'm a robot. I'm a robot. I'm a robot..."* *(Repeats infinitely)*<br> | **"Output: A robot is a robot that is designed to do something. It is the type of robot that can be programmed..."** *(Provides structural text)*<br> |
| **"Describe the role of the Federal Reserve."**<br> | *"The Federal Reserve is a central bank that is responsible for... The Federal Reserve is a central bank that is responsible for..."* *(Stuck in a sentence loop)*<br> | **"Output: The Federal Reserve serves as the central bank of the United States."** *(Provides a direct, concise summary)*<br> |
| **"How do tariffs influence international trade?"**<br> | *"### What is the impact of tariffs on international trade? ### What is the impact of tariffs on international trade?..."* *(Repeats the prompt question)*<br> | **"The tariff system allows for the protection of tariffs... Examples include the United States tariff on gasoline..."** *(Attempts real contextual explanation)*<br> |

---

## Project Architecture

Dolly-15k + FinQA
↓
Preprocessing
↓
GPT-2 + LoRA
↓
Evaluation
↓
Comparison Results
---

## Pipeline & Implementation Details

### Data Ingestion (`prepdata.py`)

Normalizes Dolly and FinQA datasets into a unified template. To accommodate local memory restraints, long financial documents are strategically truncated, and example counts are capped to build a balanced, stable training dataset.

### Parameter-Efficient Training (`tune.py`)

* **Targeting:** Applies LoRA parameters exclusively onto the query and value attention projection blocks (`c_attn`) to reduce training overhead.
* **Memory Management:** Leverages micro-batching (Batch size: 2) combined with gradient accumulation steps to maximize local hardware capability.

---

## Setup & Execution

### Installation

Ensure you have Python 3.11+ installed, then run:

```bash
git clone https://github.com/Celsius273-web/Fine_Tuning_GPT2_with_Dolly_and_FinQA
cd Fine_Tuning_GPT2_with_Dolly_and_FinQA
pip install -r requirements.txt

```

### Running the Project

1. **Preprocess Data:** `python prepdata.py`
2. **Train Adapter:** `python tune.py`
3. **Run Benchmarks:** `python inference.py`
