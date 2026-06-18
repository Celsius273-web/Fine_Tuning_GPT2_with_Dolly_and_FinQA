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
