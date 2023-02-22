
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B", bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')