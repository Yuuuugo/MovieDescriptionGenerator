from torch import nn
from transformers import Trainer



class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        prompt_input_ids = inputs["prompt_input_ids"]
        prompt_attention_mask = inputs["prompt_attention_mask"]
        completion_input_ids = inputs["completion_input_ids"]
        completion_attention_mask = inputs["completion_attention_mask"]

        loss = model(prompt_input_ids,prompt_attention_mask).loss
        

