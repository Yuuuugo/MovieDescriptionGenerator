from process.process_dataset import PlotGeneratorDataset, split_dataset
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForCausalLM
import torch
import torch.nn as nn
from transformers.trainer_pt_utils import get_parameter_names

import bitsandbytes as bnb


def optim(model, training_args):
    decay_parameters = get_parameter_names(model, [nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if n not in decay_parameters
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer_kwargs = {
        "betas": (training_args.adam_beta1, training_args.adam_beta2),
        "eps": training_args.adam_epsilon,
    }
    optimizer_kwargs["lr"] = training_args.learning_rate
    adam_bnb_optim = bnb.optim.Adam8bit(
        optimizer_grouped_parameters,
        betas=(training_args.adam_beta1, training_args.adam_beta2),
        eps=training_args.adam_epsilon,
        lr=training_args.learning_rate,
    )
    return adam_bnb_optim


torch.cuda.empty_cache()


if __name__ == "__main__":
    dataset = PlotGeneratorDataset()
    train_ds, val_ds = split_dataset(dataset)
    tokenizer = dataset.tokenizer

    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
    model.resize_token_embeddings(len(tokenizer))
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    use_cache = False

    args = TrainingArguments(
        output_dir="results/",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        gradient_checkpointing=True,
        num_train_epochs=30,
        weight_decay=0.1,
        learning_rate=5e-4,
        report_to="none",  # disable wandb
        fp16=True,
    )

    adam_bnb_optim = optim(model, args)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        data_collator=data_collator,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        optimizers=(adam_bnb_optim, None),
    )

    trainer.train()
    trainer.save_model("model/saved_model")
