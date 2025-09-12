import argparse, torch

from sympy import false
from transformers import AutoTokenizer, AutoModelForCausalLM

def get_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def main(input_text):

    model_path = 'model/gpt2_lora_finetuned'

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model.to(device)


    total, trainable = get_model_parameters(model)
    print(f"Total params: {total}, Trainable: {trainable}")
    prompt = f"Instruction: {input_text}\nInput: \nOutput:"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=1024)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model.eval()
    with torch.no_grad():
        output = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=200,
            do_sample=False,
            temperature=0.7, top_k=50, top_p=0.95
        )
    gen_ids = output[0][inputs["input_ids"].shape[-1]:]
    reply = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    print(reply)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text using a trained model.")
    parser.add_argument('input_text', type=str, help="The input text to generate from.")
    args = parser.parse_args()
    main(args.input_text)