import os
import time
import torch
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import L1Loss, BCEWithLogitsLoss
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
                 # Loss parameters
                 r1_gamma=10,
                 pcp_ratio=1.0,
                 l1_ratio=1.0,
                 bce_ratio=1.0,
                 # Training parameters
                 epochs=100,
                 gradient_accumulation=1,
                 eval_steps=500,
                 checkpoint_dir="./ckpt",
                 # WandB config
                 use_wandb=True,
                 project_name="CoReaP",
                 run_name=None):
        
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.g_optim = g_optimizer
        self.d_optim = d_optimizer
        self.device = device
        
        # Loss configurations
        self.r1_gamma = r1_gamma
        self.pcp_ratio = pcp_ratio
        self.l1_ratio = l1_ratio
        self.bce_ratio = bce_ratio
        self.pcp = PerceptualLoss(layer_weights=dict(conv4_4=1/4, conv5_4=1/2)).to(device)
        
        # Training configurations
        self.epochs = epochs
        self.gradient_accumulation = gradient_accumulation
        self.eval_steps = eval_steps
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Logging
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(project=project_name, name=run_name or f"run_{time.strftime('%Y%m%d_%H%M%S')}")
        
        # Loss functions
        self.l1_loss = L1Loss()
        self.bce_loss = BCEWithLogitsLoss()

    def compute_loss(self, batch):
        # 데이터 준비
        img = batch['img'].to(self.device)
        mask_img = batch['mask_img'].to(self.device)
        edge = batch['edge'].to(self.device)
        mask = batch['mask'].to(self.device)
        
        # Generator 포워드 패스
        z = torch.randn(img.size(0), 512).to(self.device)

        print(mask_img.shape, mask.shape, z.shape, edge.shape, img.shape)
        gen_img, gen_img_stg1, high_freq = self.generator(
            mask_img, mask, z, None, return_stg1=True, edge=edge, line=edge
        )
        
        # Discriminator 포워드 패스
        with torch.no_grad():
            real_logits, _ = self.discriminator(img, mask, img, None)
            fake_logits, _ = self.discriminator(gen_img.detach(), mask, gen_img_stg1.detach(), None)
        
        # Generator Loss 계산
        g_loss_gan = torch.nn.functional.softplus(-fake_logits).mean()
        g_loss_l1 = self.l1_loss(gen_img, img)
        pcp_loss, _ = self.pcp(gen_img, img)
        bce_loss = self.bce_loss(high_freq, edge)
        g_loss = g_loss_gan + self.pcp_ratio*pcp_loss + self.l1_ratio*g_loss_l1 + self.bce_ratio*bce_loss
        
        # Discriminator Loss 계산
        d_loss_real = torch.nn.functional.softplus(-real_logits).mean()
        d_loss_fake = torch.nn.functional.softplus(fake_logits).mean()
        d_loss = d_loss_real + d_loss_fake
        
        # R1 Regularization
        if self.r1_gamma > 0:
            real_img = img.detach().requires_grad_(True)
            real_logits, _ = self.discriminator(real_img, mask, real_img)
            r1_grads = torch.autograd.grad(
                outputs=real_logits.sum(),
                inputs=real_img,
                create_graph=True
            )[0]
            r1_penalty = r1_grads.pow(2).sum([1,2,3]).mean()
            d_loss += self.r1_gamma * 0.5 * r1_penalty
        
        return {
            "g_loss": g_loss,
            "d_loss": d_loss,
            "gen_img": gen_img.detach(),
            "real_img": img.detach()
        }

    def train_epoch(self, epoch):
        self.generator.train()
        self.discriminator.train()
        
        total_g_loss = 0.0
        total_d_loss = 0.0
        accum_step = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        for batch_idx, batch in enumerate(progress_bar):
            # Discriminator 업데이트
            self.discriminator.requires_grad_(True)
            self.generator.requires_grad_(False)
            
            losses = self.compute_loss(batch)
            scaled_d_loss = losses["d_loss"] / self.gradient_accumulation
            scaled_d_loss.backward(retain_graph=True)
            
            # Generator 업데이트
            self.discriminator.requires_grad_(False)
            self.generator.requires_grad_(True)
            
            scaled_g_loss = losses["g_loss"] / self.gradient_accumulation
            scaled_g_loss.backward()
            
            # Gradient Accumulation
            accum_step += 1
            if accum_step % self.gradient_accumulation == 0:
                self.d_optim.step()
                self.g_optim.step()
                self.d_optim.zero_grad()
                self.g_optim.zero_grad()
            
            # Loss 누적
            total_g_loss += losses["g_loss"].item()
            total_d_loss += losses["d_loss"].item()
            
            # 로깅
            if self.use_wandb and (batch_idx % 10 == 0):
                wandb.log({
                    "train/g_loss": losses["g_loss"].item(),
                    "train/d_loss": losses["d_loss"].item(),
                    "epoch": epoch,
                    "step": epoch * len(self.train_loader) + batch_idx
                })
                
            progress_bar.set_postfix({
                "g_loss": f"{total_g_loss/(batch_idx+1):.4f}",
                "d_loss": f"{total_d_loss/(batch_idx+1):.4f}"
            })
            
            # 평가 실행
            if (batch_idx + 1) % self.eval_steps == 0:
                eval_metrics = self.evaluate()
                if self.use_wandb:
                    wandb.log(eval_metrics)
                
        return total_g_loss / len(self.train_loader), total_d_loss / len(self.train_loader)

    def evaluate(self):
        self.generator.eval()
        total_loss = 0.0
        metrics = {}
        
        with torch.no_grad():
            for batch in self.val_loader:
                losses = self.compute_loss(batch)
                total_loss += (losses["g_loss"] + losses["d_loss"]).item()
                
                # 여기에 추가 메트릭 계산 로직 구현
                # 예: compute_metrics(losses["gen_img"], losses["real_img"])
                
        avg_loss = total_loss / len(self.val_loader)
        metrics.update({"eval/loss": avg_loss})
        
        # 생성 이미지 시각화
        if self.use_wandb:
            gen_images = wandb.Image(losses["gen_img"][:4].cpu())
            real_images = wandb.Image(losses["real_img"][:4].cpu())
            wandb.log({"generated": gen_images, "real": real_images})
            
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
        for epoch in range(self.epochs):
            start_time = time.time()
            avg_g_loss, avg_d_loss = self.train_epoch(epoch)
            
            # 에폭 종료 시 로깅
            if self.use_wandb:
                wandb.log({
                    "epoch/avg_g_loss": avg_g_loss,
                    "epoch/avg_d_loss": avg_d_loss,
                    "epoch/time": time.time()-start_time
                })
                
            # 체크포인트 저장
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch + 1)

        # 학습 종료 후 최종 모델 저장
        torch.save(self.generator.state_dict(), os.path.join(self.checkpoint_dir, "final_generator.pt"))
        torch.save(self.discriminator.state_dict(), os.path.join(self.checkpoint_dir, "final_discriminator.pt"))