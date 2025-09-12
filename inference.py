import torch, json
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import logging


def get_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def main():
    logging.set_verbosity_error()
    with open("test.json", "r") as f:
        try:
            promptList = json.load(f)
        except json.JSONDecodeError:
            print("Test prompts didn't load")

    model_path = 'model/gpt2_lora_finetuned'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path)

    reg_model_path = "gpt2/gpt2_model"
    tokenizer_reg = AutoTokenizer.from_pretrained(reg_model_path)
    tokenizer_reg.pad_token = tokenizer_reg.eos_token
    reg_model = AutoModelForCausalLM.from_pretrained(reg_model_path)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model.to(device)
    reg_model.to(device)


    total, trainable = get_model_parameters(model)
    print(f"Total params: {total}, Trainable: {trainable}")
    total, trainable = get_model_parameters(reg_model)
    print(f"Total params reg model: {total}, Trainable: {trainable}")
    results = []
    for query in promptList:
        prompt = f"### Instruction:\n{query}\n\n### Response:\n"

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=1024)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        model.eval()
        with torch.no_grad():
            output = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=200,
                do_sample=True, #i tried false but this way it doesnt always choose the highest prob token - a little more creative
                temperature=0.7, top_k=50, top_p=0.95
            )
        gen_ids = output[0][inputs["input_ids"].shape[-1]:]
        fine_tuned = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        inputs2 = tokenizer_reg(prompt, return_tensors="pt", truncation=True, padding=True, max_length=1024)
        inputs2 = {k: v.to(device) for k, v in inputs2.items()}
        reg_model.eval()
        with torch.no_grad():
            reg_output = reg_model.generate(
                input_ids=inputs2["input_ids"],
                attention_mask=inputs2["attention_mask"],
                max_new_tokens=200,
                do_sample=False,
                temperature=0.7, top_k=50, top_p=0.95
            )
        reg_gen_ids = reg_output[0][inputs2["input_ids"].shape[-1]:]
        regular = tokenizer_reg.decode(reg_gen_ids, skip_special_tokens=True).strip()
        results.append({
            "prompt": query,
            "fine_tuned": fine_tuned,
            "regular": regular
        })

    with open("model_comparison.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()