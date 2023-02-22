from torch.utils.data import Dataset, random_split
from transformers import AutoTokenizer
import pandas as pd
import torch


class PlotGeneratorDataset(Dataset):
    def __init__(self,tokenizer,prompt_max_len,completion_max_len,df):
        self.tokenizer = tokenizer
        self.prompt_max_len = prompt_max_len
        self.completion_max_len = completion_max_len
        self.df = df
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        prompt = self.df.prompt[index]
        completion = self.df.completion[index]
        encode_prompt = self.tokenizer('<|startoftext|>' + prompt + '<|endoftext|>', truncation=True, max_length=self.prompt_max_len, padding="max_length")
        encode_completion = self.tokenizer('<|startoftext|>' + completion + '<|endoftext|>', truncation=True, max_length=self.completion_max_len, padding="max_length")

        return {
            "prompt_input_ids": torch.tensor(encode_prompt['input_ids'], dtype=torch.long),
            "prompt_attention_mask": torch.tensor(encode_prompt['attention_mask'], dtype=torch.long),
            "completion_input_ids": torch.tensor(encode_completion['input_ids'], dtype=torch.long),
            "completion_attention_mask": torch.tensor(encode_completion['attention_mask'], dtype=torch.long)
        }

    

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B", bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')
    df = pd.read_csv('../data/processed.csv', sep=',')
    ds = PlotGeneratorDataset(tokenizer, 15, 100, df)
    print(ds[0])