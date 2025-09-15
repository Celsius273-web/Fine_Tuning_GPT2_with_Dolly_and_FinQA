GPT-2 Fine-Tuning with LoRA

This project fine-tunes GPT-2 with LoRA adapters on multiple instruction-style datasets, demonstrating noticeable improvements in response quality compared to the base model. It includes scripts that prep data, train, do inference, and output the different responses from the model. 

Overview

The project fine-tunes GPT-2 with LoRA adapters on two datasets:

Dolly 15k: Instruction-following dataset from Databricks.
FinQA: Financial reasoning question-answering dataset.

Getting Started

Clone this repository and set up a Python virtual environment:
pip install -r requirements.txt
git clone https://github.com/Celsius273-web/Fine_Tuning_GPT2_with_Dolly_and_FinQA
Also, I reccomend using Python 3.11 as it works well with all the libraries and models.

Make sure your data directory contains:

train.jsonl
val.jsonl

As these files are generated with information from datasets that is organized by prepdata.py and put into the jsonl files. 

Datasets and Models

Datasets:

Dolly 15k: Instruction-following examples for GPT-2 fine-tuning
Source: https://huggingface.co/databricks/databricks-dolly-15k


FinQA: Financial reasoning examples for GPT-2 fine-tuning
Source: https://huggingface.co/datasets/PTPReasoning/finqa


Base Model:

GPT-2
Source: https://huggingface.co/gpt2

Usage: Base model for LoRA fine-tuning. The fine-tuned LoRA adapter is stored in model/gpt2_lora_finetuned/.

Usage

To prepare the data for the llm to take in use prepdata.py

To train or fine-tune: python3 tune.py

To compare base vs fine-tuned models use: python3 inference.py 

After fine-tuning, compare the performance of the base GPT-2 model and the fine-tuned model by giving each model the prompts (prompts I tested with are in test.json) and the results can be seen in the file model_comparison.json.

Hardware Requirements

CPU: Depends on settings - I did it with 16GB of RAM but 8 might work. I do reccommend looking at the finetune.py to control how much computation will be done (this will impact the model's "learning").
GPU: Recommended for faster training, but not required
I used LoRA fine-tuning as it is lightweight, so full model weights are not needed for sharing. 

Results

After fine-tuning, the model shows a clear improvement compared to the base GPT-2. The fine-tuned version is generally better at answering questions directly instead of simply repeating the prompt. For example, when asked to explain “What is a reverse merger?”, the base model mostly repeated the question whereas the fine-tuned model produced a decent definition. In creative prompts like “Create a Poem about the Ocean,” the fine-tuned model generated something instead of looping phrases. The model isn't fantastic as responses can be repetitive or shallow but the difference shows that lightweight fine-tuning with LoRA adapters can noticeably improve GPT-2's responses.

If you want to Expand this Project

You can include additional datasets by:

Adding functionality in prepdata.py to load and organize more data.
Then reloading train.jsonl and val.jsonl with new data from prepdata.py.

You can play around with finetune.py to change the finetuned model. It is designed with the consideration of computational power.
Or you can change the model being used, though it will require some changes and more computing power for models with more parameters.

Citing

Here is the official citation of the models and data set. Please cite the datasets and base model if you are going to use this repository. 

Dolly 15k: Databricks Dolly 15k (https://huggingface.co/databricks/databricks-dolly-15k)
@online{DatabricksBlog2023DollyV2,
    author    = {Mike Conover and Matt Hayes and Ankit Mathur and Jianwei Xie and Jun Wan and Sam Shah and Ali Ghodsi and Patrick Wendell and Matei Zaharia and Reynold Xin},
    title     = {Free Dolly: Introducing the World's First Truly Open Instruction-Tuned LLM},
    year      = {2023},
    url       = {https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm},
    urldate   = {2023-06-30}
}

FinQA: PTPReasoning/FinQA 
@inproceedings{chen2021finqa,
          title={FinQA: A Dataset of Numerical Reasoning over Financial Data},
          author={Chen, Zhiyu and Chen, Wenhu and Smiley, Charese and Shah, Sameena and Borova, Iana and Langdon, Dylan and Moussa, Reema and Beane, Matt and Huang, Ting-Hao and Routledge, Bryan R and others},
          booktitle={Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing},
          pages={3697--3711},
          year={2021}
        }


GPT-2: OpenAI GPT-2 
@article{radford2019language,
  title={Language Models are Unsupervised Multitask Learners},
  author={Radford, Alec and Wu, Jeff and Child, Rewon and Luan, David and Amodei, Dario and Sutskever, Ilya},
  year={2019}
}


License

This project is licensed under the MIT License. 

Final Note: Have fun! 
