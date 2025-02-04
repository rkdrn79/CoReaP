import os
import time
import torch
import wandb
from tqdm import tqdm
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import L1Loss, BCEWithLogitsLoss
from torch.cuda.amp import autocast

from src.utils.losses.pcp import PerceptualLoss
from src.utils.metrics import compute_metrics

class CoReaPTrainer:
    def __init__(self, 
                 generator, 
                 discriminator,
                 train_loader,
                 val_loader,
                 g_optimizer,
                 d_optimizer,
                 device,
                 r1_gamma=10,
                 pcp_ratio=1.0,
                 l1_ratio=1.0,
                 bce_ratio=1.0,
                 epochs=100,
                 gradient_accumulation=1,
                 eval_steps=500,
                 checkpoint_dir="./ckpt",
                 use_wandb=True,
                 project_name="CoReaP",
                 run_name=None,
                 args=None):
        self.args = args
        self.generator = generator
        self.discriminator = discriminator
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.g_optim = g_optimizer
        self.d_optim = d_optimizer
        self.device = device

        # Other initialization remains the same
        self.r1_gamma = r1_gamma
        self.pcp_ratio = pcp_ratio
        self.l1_ratio = l1_ratio
        self.bce_ratio = bce_ratio
        self.pcp = PerceptualLoss(layer_weights=dict(conv4_4=1/4, conv5_4=1/2)).to(device)
        self.epochs = epochs
        self.gradient_accumulation = gradient_accumulation
        self.eval_steps = eval_steps
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(project=project_name, name=run_name or f"run_{time.strftime('%Y%m%d_%H%M%S')}")
        self.l1_loss = L1Loss()
        self.bce_loss = BCEWithLogitsLoss()

    def compute_d_loss(self, batch, train=True):
        img = batch['img'].to(self.device)
        mask_img = batch['mask_img'].to(self.device)
        edge = batch['edge'].to(self.device)
        mask = batch['mask'].to(self.device)

        
        with torch.no_grad():
            z = torch.randn(img.size(0), img.size(2)).to(self.device)
            if self.args.bf16:
                z = z.to(dtype=torch.bfloat16)
            
            # bf16 autocast
            with autocast(dtype=torch.bfloat16 if self.args.bf16 else torch.float32):
                gen_img, gen_img_stg1, high_freq = self.generator(
                    mask_img, mask, z, None, return_stg1=True, edge=edge, line=None
                )

        real_logits, _ = self.discriminator(img, mask, img, None)
        fake_logits, _ = self.discriminator(gen_img.detach(), mask, gen_img_stg1.detach(), None)
        
        d_loss_real = torch.nn.functional.softplus(-real_logits).mean()
        d_loss_fake = torch.nn.functional.softplus(fake_logits).mean()
        d_loss = d_loss_real + d_loss_fake
        
        if self.r1_gamma > 0 and train:
            real_img = img.detach().requires_grad_(True)
            real_logits, _ = self.discriminator(real_img, mask, real_img, None)
            r1_grads = torch.autograd.grad(
                outputs=real_logits.sum(),
                inputs=real_img,
                create_graph=True
            )[0]
            r1_penalty = r1_grads.pow(2).sum([1,2,3]).mean()
            d_loss += self.r1_gamma * 0.5 * r1_penalty
        
        return d_loss, gen_img, gen_img_stg1, high_freq

    def compute_g_loss(self, batch):
        img = batch['img'].to(self.device)
        mask_img = batch['mask_img'].to(self.device)
        edge = batch['edge'].to(self.device)
        mask = batch['mask'].to(self.device)
        
        z = torch.randn(img.size(0), img.size(2)).to(self.device)
        if self.args.bf16:
            z = z.to(dtype=torch.bfloat16)

        # bf16 autocast
        with autocast(dtype=torch.bfloat16 if self.args.bf16 else torch.float32):
            gen_img, gen_img_stg1, high_freq = self.generator(
                mask_img, mask, z, None, return_stg1=True, edge=edge, line=None
            )
        
        with torch.no_grad():
            fake_logits, _ = self.discriminator(gen_img, mask, gen_img_stg1, None)
        
        g_loss_gan = torch.nn.functional.softplus(-fake_logits).mean()
        g_loss_l1 = self.l1_loss(gen_img, img)
        pcp_loss, _ = self.pcp(gen_img, img)
        high_freq = high_freq[:, 0:1, :, :]
        bce_loss = self.bce_loss(high_freq, edge)
        g_loss = g_loss_gan + self.pcp_ratio * pcp_loss + self.l1_ratio * g_loss_l1 + self.bce_ratio * bce_loss
        
        return g_loss, gen_img, gen_img_stg1, high_freq

    def train_epoch(self, epoch):
        self.generator.train()
        self.discriminator.train()
        total_g_loss = 0.0
        total_d_loss = 0.0
        accum_step = 0
        
        
        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch+1}")):
            # Mixed precision training
            with autocast(enabled=self.args.bf16, dtype=torch.bfloat16):
                # Update Discriminator
                self.discriminator.requires_grad_(True)
                self.generator.requires_grad_(False)
                d_loss, _, _, _ = self.compute_d_loss(batch)
                scaled_d_loss = d_loss / self.gradient_accumulation
               
                scaled_d_loss.backward()

                # Update Generator
                self.discriminator.requires_grad_(False)
                self.generator.requires_grad_(True)
                g_loss, gen_img, _, _ = self.compute_g_loss(batch)
                scaled_g_loss = g_loss / self.gradient_accumulation
                
                scaled_g_loss.backward()
            
            accum_step += 1
            if accum_step % self.gradient_accumulation == 0:
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
                
                # Optimizer step
                self.d_optim.step()
                self.g_optim.step()
                
                # Zero gradients
                self.d_optim.zero_grad()
                self.g_optim.zero_grad()
                accum_step = 0
            
            # Accumulate losses (reduce across all processes)
            total_d_loss += d_loss.item()
            total_g_loss += g_loss.item()
            
            # WandB logging only on rank 0
            if self.use_wandb and (batch_idx % 10 == 0):
                wandb.log({
                    "train/g_loss": g_loss.item(),
                    "train/d_loss": d_loss.item(),
                    "epoch": epoch,
                    "step": epoch * len(self.train_loader) + batch_idx
                })
                
            
            if (batch_idx + 1) % self.eval_steps == 0:
                eval_metrics = self.evaluate()
                if self.use_wandb == 0:
                    wandb.log(eval_metrics)
            
        return total_g_loss / len(self.train_loader), total_d_loss / len(self.train_loader)


    def evaluate(self):
        self.generator.eval()
        total_loss = 0.0

        log_images = []
        log_edges = []
        log_gen_images = []
        log_high_freq = []
        with torch.no_grad():
            cnt = 0
            for batch in tqdm(self.val_loader, desc="Evaluation"):
                g_loss, gen_img, _, high_freq = self.compute_g_loss(batch)
                total_loss += g_loss.item()

                # 첫번째 이미지만 로깅
                log_images.append(batch['img'][0].cpu())
                log_edges.append(batch['edge'][0].cpu())
                log_gen_images.append(gen_img[0].cpu())
                log_high_freq.append(high_freq[0].cpu())

                cnt += 1

                if cnt == 10:
                    break

        if self.use_wandb:
            # Log loss
            wandb.log({
                "eval/loss": total_loss / 10
            })

            # TODO : Log Metrics
            metrics = None #compute_metrics(log_images, log_gen_images)

            # Log images
            wandb.log({
                "eval/images": [wandb.Image(img) for img in log_images],
                "eval/edges": [wandb.Image(edge) for edge in log_edges],
                "eval/gen_images": [wandb.Image(gen_img) for gen_img in log_gen_images],
                "eval/high_freq": [wandb.Image(high_freq) for high_freq in log_high_freq]
            })
        return metrics

    def save_checkpoint(self, epoch):
        checkpoint = {
            "generator": self.generator.state_dict(),
            "discriminator": self.discriminator.state_dict(),
            "g_optimizer": self.g_optim.state_dict(),
            "d_optimizer": self.d_optim.state_dict(),
            "epoch": epoch
        }
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt"))

    def train(self):
        for epoch in tqdm(range(self.epochs), desc="Training"):
            start_time = time.time()
            avg_g_loss, avg_d_loss = self.train_epoch(epoch)
            if self.use_wandb:
                wandb.log({
                    "epoch/avg_g_loss": avg_g_loss,
                    "epoch/avg_d_loss": avg_d_loss,
                    "epoch/time": time.time()-start_time
                })
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch + 1)
        torch.save(self.generator.state_dict(), os.path.join(self.checkpoint_dir, "final_generator.pt"))
        torch.save(self.discriminator.state_dict(), os.path.join(self.checkpoint_dir, "final_discriminator.pt"))
