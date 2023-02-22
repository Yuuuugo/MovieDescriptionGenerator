import torch 
import torch.nn as nn


def generate_print(model, tokenizer, prompt_text, max_length=100, num_return_sequences=1, device = 'cpu'):
    input_ids = tokenizer(prompt_text, return_tensors='pt')["input_ids"].to(device)
    input_ids = input_ids.to('cuda')
    sample_outputs = model.generate(
    input_ids,
    do_sample=True, 
    max_length=50, 
    top_k=50, 
    top_p=0.95, 
    num_return_sequences=3
    )
    print("Output:\n" + 100 * '-')
    for i, sample_output in enumerate(sample_outputs):
          print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))

