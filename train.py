import os
import warnings
warnings.filterwarnings("ignore")

import torch
from transformers import TrainingArguments
import random
import numpy as np
import wandb
import torch

from arguments import get_arguments

from src.dataset.get_dataset import get_dataset
from src.model.get_model import get_model

from src.trainer import CoReaPTrainer
from src.utils.metrics import compute_metrics

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main(args):
    set_seed(args.seed)
    print(args)

    # Load dataset
    train_dataset, val_dataset, data_collator = get_dataset(args)

    # Load Model 
    model = get_model(args)

    # wandb
    wandb.init(project='CoReaP', name=args.save_dir)

    training_args = TrainingArguments(
        output_dir=f"./model/{args.save_dir}",
        # eval
        eval_strategy='steps',
        eval_steps=args.eval_steps,
        logging_steps=10,
        metric_for_best_model="metric", # metric for best model
        # save
        save_strategy="epoch",
        save_total_limit=None,
        save_steps=args.eval_steps,
        remove_unused_columns=False,
        # training
        per_device_train_batch_size=args.per_device_train_batch_size, 
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        learning_rate=args.learning_rate,
        # utils
        report_to="wandb",
        dataloader_num_workers=0,
        bf16 = args.bf16,
    )

    trainer = CoReaPTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    print(args)

    trainer.train()

if __name__=="__main__":

    args = get_arguments()
    main(args)