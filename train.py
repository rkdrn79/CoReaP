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

from src.datasets.get_dataset import get_dataset
from src.model.get_model import get_model

from src.CoReaPTrainer import CoReaPTrainer

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

    # 데이터 로더 생성
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.per_device_train_batch_size, 
        shuffle=True, 
        collate_fn=data_collator
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=args.per_device_eval_batch_size, 
        shuffle=False, 
        collate_fn=data_collator
    )

    # Load Model
    generator, discriminator = get_model(args)
    g_optimizer = torch.optim.Adam(
        generator.parameters(), 
        lr=args.learning_rate, 
        betas=(0.5, 0.999)
    )
    d_optimizer = torch.optim.Adam(
        discriminator.parameters(), 
        lr=args.learning_rate, 
        betas=(0.5, 0.999)
    )

    # wandb 초기화
    wandb.init(project='CoReaP', name=args.save_dir)

    # GANTrainer 초기화
    trainer = CoReaPTrainer(
        generator=generator,
        discriminator=discriminator,
        train_loader=train_loader,
        val_loader=val_loader,
        g_optimizer=g_optimizer,
        d_optimizer=d_optimizer,
        device=device,
        # Loss parameters
        r1_gamma=args.r1_gamma,
        pcp_ratio=args.pcp_ratio,
        l1_ratio=args.l1_ratio,
        bce_ratio=args.bce_ratio,
        # Training parameters
        epochs=args.num_train_epochs,
        gradient_accumulation=args.gradient_accumulation_steps,
        eval_steps=args.eval_steps,
        checkpoint_dir=f"./ckpt/{args.save_dir}",
        # WandB config
        use_wandb=True,
        project_name="CoReaP",
        run_name=args.save_dir
    )

    print(args)

    # 학습 시작
    trainer.train()

if __name__=="__main__":

    args = get_arguments()
    main(args)