import torch 
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM


def generate_print(model, tokenizer, prompt_text, max_length=100, num_return_sequences=1, device = 'cpu'):
    input_ids = tokenizer(prompt_text, return_tensors='pt')["input_ids"].to(device)
    input_ids = input_ids.to('cuda')
    sample_outputs = model.generate(
    input_ids,
    do_sample=True, 
    max_length=max_length, 
    top_k=50, 
    top_p=0.95, 
    num_return_sequences=num_return_sequences
    )
    print("Output:\n" + 100 * '-')
    for i, sample_output in enumerate(sample_outputs):
          print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))



if __name__ == "__main__":
    model_checkpoint = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForCausalLM.from_pretrained(model_checkpoint)
    model.to('cuda')
    prompt_text = "The last of us :"
    generate_print(model, tokenizer, prompt_text)


