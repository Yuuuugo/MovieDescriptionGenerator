from transformers import TrainingArguments, IntervalStrategy,  AutoModelForCausalLM






if __name__ == '__main__':
    training_args = TrainingArguments(output_dir='./results', num_train_epochs=4.3, logging_steps=50, save_strategy=IntervalStrategy.NO,
                                  per_device_train_batch_size=15, per_device_eval_batch_size=15, warmup_steps=50,
                                  weight_decay=0.01, logging_dir='./logs', fp16=True, deepspeed='./ds_config_gpt_neo_27.json')
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B", pad_token_id=tokenizer.eos_token_id)
