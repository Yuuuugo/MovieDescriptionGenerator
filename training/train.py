from transformers import (
    TrainingArguments,
    IntervalStrategy,
    AutoModelForCausalLM,
    Trainer,
    AutoTokenizer,
)
from process.process_dataset import split_dataset


if __name__ == "__main__":
    training_args = TrainingArguments(
        output_dir="../results",
        num_train_epochs=4.3,
        logging_steps=50,
        save_strategy=IntervalStrategy.NO,
        per_device_train_batch_size=15,
        per_device_eval_batch_size=15,
        warmup_steps=50,
        weight_decay=0.01,
        logging_dir="./logs",
        fp16=True,
        deepspeed="./config_gpt2.json",
        report_to="wandb",
    )

    model = AutoModelForCausalLM.from_pretrained("gpt2-medium")
    tokenizer = AutoTokenizer.from_pretrained(
        "gpt2-medium",
        bos_token="<|startoftext|>",
        eos_token="<|endoftext|>",
        pad_token="<|pad|>",
    )
    model.resize_token_embeddings(len(tokenizer))
    train_ds, val_ds = split_dataset()
    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_ds, eval_dataset=val_ds
    )
