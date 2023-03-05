from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoTokenizer, GPT2LMHeadModel, AutoConfig
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForCausalLM


def tokenize(element):
    outputs = tokenizer(
        element["text"],
        truncation=True,
        padding="max_length",
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True,
    )
    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        if length == context_length:
            input_batch.append(input_ids)
    return {"input_ids": input_batch}


if __name__ == "__main__":
    dataset = load_dataset("csv", data_files="data/processed.csv", split="train")
    dataset = dataset.train_test_split(test_size=0.2)
    context_length = 512
    tokenizer = AutoTokenizer.from_pretrained(
        "EleutherAI/gpt-neo-2.7B", pad_token="[PAD]"
    )

    tokenized_datasets = dataset.map(
        tokenize, batched=True, remove_columns=dataset["train"].column_names
    )

    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B")
    model.resize_token_embeddings(len(tokenizer))
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    args = TrainingArguments(
        output_dir="results/",
        per_device_train_batch_size=3,
        per_device_eval_batch_size=1,
        evaluation_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        gradient_accumulation_steps=1,
        num_train_epochs=30,
        weight_decay=0.1,
        warmup_steps=1_000,
        lr_scheduler_type="cosine",
        learning_rate=5e-4,
        save_steps=5_000,
        report_to="none",  # disable wandb
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
    )

    trainer.train()
    trainer.save_model("model/saved_model")
