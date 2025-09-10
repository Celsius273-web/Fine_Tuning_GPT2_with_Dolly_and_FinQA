from datasets import load_dataset
from transformers import AutoTokenizer


def run():
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    dolly = load_dataset("databricks/databricks-dolly-15k")
    oa = load_dataset("OpenAssistant/oasst1")

    clean_dolly = dolly.filter(lambda x: x['response'] is not None and not x.get('unsafe', False))
    oa = oa.filter(lambda x: x['lang'] == 'en')
    clean_oa = oa.filter(lambda x: x['text'] is not None and not x.get('unsafe', False))
    # norm_dolly = clean_dolly.map(lambda x: {"formatted_chat": tokenizer.apply_chat_template(
    #         f"User: {x['instruction']}\nContext: {x['context']}\nModel: {x['response']}",
    #         tokenize=False, add_generation_prompt=False)
    # })
    #
    # norm_oa = clean_oa.map(lambda x: {"formatted_chat": tokenizer.apply_chat_template(
    #         f"User: {x['text']}\nModel: {x['role']}",  # Assuming 'role' indicates the model's response
    #         tokenize=False,
    #         add_generation_prompt=False
    #     )
    # })
    print(clean_dolly[0])
    print(clean_oa[0])

def main():
    run()

if __name__ == "__main__":
    main()