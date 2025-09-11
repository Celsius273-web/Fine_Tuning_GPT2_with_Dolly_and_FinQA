from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

def main():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model_dir = "Qwen/Qwen3-32B"
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    model.config.use_cache = False
    model.config.pretraining_tp = 1
if __name__ == "__main__":
    main()