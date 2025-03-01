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
from src.utils.losses.focal import FocalLoss
# from src.utils.metrics import compute_metrics  # 필요시 사용

class CoReaPTrainer:
    def __init__(self, 
                 generator, 
                 discriminator,
                 train_loader,
                 val_loader,
                 g_optimizer,
                 d_optimizer,
                 device,
                 pcp_gamma=1.0,
                 r1_gamma=10,
                 high_freq_gamma=1.0,
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

        self.r1_gamma = r1_gamma
        self.pcp_gamma = pcp_gamma
        self.high_freq_gamma = high_freq_gamma

        # Perceptual Loss (PCP)
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
        self.focal_loss = FocalLoss()

    def compute_losses(self, batch, train=True):
        """
        한 번의 generator forward pass로 Generator 및 Discriminator 손실을 함께 계산합니다.
        """
        # 데이터 로딩 (c를 포함)
        img = batch['img'].to(self.device)
        mask_img = batch['mask_img'].to(self.device)
        edge = batch['edge'].to(self.device)
        mask_edge_img = batch['mask_edge_img'].to(self.device)
        mask = batch['mask'].to(self.device)
        c = None

        # 랜덤 벡터 z 생성 (모델에 맞게 차원 확인)
        z = torch.randn(img.size(0), img.size(2)).to(self.device)
        if self.args.bf16:
            z = z.to(dtype=torch.bfloat16)

        with autocast(dtype=torch.bfloat16 if self.args.bf16 else torch.float32):
            # generator forward pass 한 번 실행
            gen_img, gen_img_stg1, high_freq = self.generator(
                mask_img, mask, z, None, return_stg1=True, edge=mask_edge_img, line=None
            )

            # Generator 손실용: Discriminator에 그대로 통과 (generator의 gradient 흐름 유지)
            fake_logits_g, fake_logits_stg1_g = self.discriminator(gen_img, mask, gen_img_stg1, c)
            loss_Gmain = torch.nn.functional.softplus(-fake_logits_g)
            loss_Gmain_stg1 = torch.nn.functional.softplus(-fake_logits_stg1_g)
            pcp_loss, _ = self.pcp(gen_img, img)
            # High Frequency Loss: 첫 번째 채널는 edge, 두 번째 채널은 0 텐서와 비교
            high_freq_loss_1 = self.focal_loss(high_freq[:, 0], edge) + self.l1_loss(high_freq[:, 0], edge)
            high_freq_loss_2 = self.focal_loss(high_freq[:, 1], torch.zeros_like(edge)) + self.l1_loss(high_freq[:, 1], torch.zeros_like(edge))
            high_freq_loss = high_freq_loss_1 + high_freq_loss_2

            g_loss = loss_Gmain.mean() + loss_Gmain_stg1.mean() + self.pcp_gamma * pcp_loss + self.high_freq_gamma * high_freq_loss

            # Discriminator 손실용: generator 출력은 detach하여 generator의 gradient 차단
            fake_logits_d, fake_logits_stg1_d = self.discriminator(gen_img.detach(), mask, gen_img_stg1.detach(), c)
            loss_Dgen = torch.nn.functional.softplus(fake_logits_d)
            loss_Dgen_stg1 = torch.nn.functional.softplus(fake_logits_stg1_d)

            # 실제 이미지에 대한 Discriminator 예측
            real_logits, real_logits_stg1 = self.discriminator(img, mask, img, c)
            loss_Dreal = torch.nn.functional.softplus(-real_logits)
            loss_Dreal_stg1 = torch.nn.functional.softplus(-real_logits_stg1)

            # R1 정규화: 실제 이미지에 대한 gradient 계산
            if train:
                img_tmp = img.detach().requires_grad_(True)
                real_logits_reg, real_logits_reg_stg1 = self.discriminator(img_tmp, mask, img_tmp, c)
                r1_grads = torch.autograd.grad(outputs=real_logits_reg.sum(), inputs=img_tmp, create_graph=True)[0]
                r1_penalty = r1_grads.pow(2).sum([1, 2, 3]).mean()
                r1_loss = r1_penalty * (self.r1_gamma / 2)
            else:
                r1_loss = torch.tensor(0.0).to(img.device)

            d_loss = loss_Dgen.mean() + loss_Dgen_stg1.mean() + loss_Dreal.mean() + loss_Dreal_stg1.mean() + r1_loss

        losses = {
            "g_loss": g_loss,
            "loss_Gmain": loss_Gmain.mean(),
            "loss_Gmain_stg1": loss_Gmain_stg1.mean(),
            "pcp_loss": pcp_loss,
            "high_freq_loss": high_freq_loss,
            "d_loss": d_loss,
            "loss_Dgen": loss_Dgen.mean(),
            "loss_Dgen_stg1": loss_Dgen_stg1.mean(),
            "loss_Dreal": loss_Dreal.mean(),
            "loss_Dreal_stg1": loss_Dreal_stg1.mean(),
            "r1_loss": r1_loss
        }
        outputs = {
            "gen_img": gen_img,
            "gen_img_stg1": gen_img_stg1,
            "high_freq": high_freq
        }
        return g_loss, d_loss, losses, outputs

    def train_epoch(self, epoch):
        self.generator.train()
        self.discriminator.train()
        total_g_loss = 0.0
        total_d_loss = 0.0
        accum_step = 0

        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch+1}")):
            # 한 배치당 forward pass 1회로 두 손실을 모두 계산
            with autocast(enabled=self.args.bf16, dtype=torch.bfloat16 if self.args.bf16 else torch.float32):
                g_loss, d_loss, losses, outputs = self.compute_losses(batch)
            # gradient accumulation에 따라 backward 수행 (두 네트워크 업데이트를 분리)
            accum_step += 1

            # Discriminator 업데이트 (generator forward 결과는 detach되어 있으므로 문제없음)
            self.discriminator.requires_grad_(True)
            self.generator.requires_grad_(False)
            d_loss.backward(retain_graph=True)

            # Generator 업데이트
            self.discriminator.requires_grad_(False)
            self.generator.requires_grad_(True)
            g_loss.backward()

            if accum_step % self.gradient_accumulation == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)

                # Optimizer step 및 gradient 초기화
                self.d_optim.step()
                self.g_optim.step()
                self.d_optim.zero_grad()
                self.g_optim.zero_grad()
                accum_step = 0

            total_d_loss += d_loss.item()
            total_g_loss += g_loss.item()

            # WandB 로깅 (10 iter마다)
            if self.use_wandb and (batch_idx % 10 == 0):
                wandb.log({
                    "train/g_loss": g_loss.item(),
                    "train/loss_Gmain": losses["loss_Gmain"].item(),
                    "train/loss_Gmain_stg1": losses["loss_Gmain_stg1"].item(),
                    "train/pcp_loss": losses["pcp_loss"].item(),
                    "train/high_freq_loss": losses["high_freq_loss"].item(),
                    "train/d_loss": d_loss.item(),
                    "train/loss_Dgen": losses["loss_Dgen"].item(),
                    "train/loss_Dgen_stg1": losses["loss_Dgen_stg1"].item(),
                    "train/loss_Dreal": losses["loss_Dreal"].item(),
                    "train/loss_Dreal_stg1": losses["loss_Dreal_stg1"].item(),
                    "train/r1_loss": losses["r1_loss"].item(),
                    "epoch": epoch,
                    "step": epoch * len(self.train_loader) + batch_idx
                })

            if (batch_idx + 1) % self.eval_steps == 0:
                self.evaluate()

        return total_g_loss / len(self.train_loader), total_d_loss / len(self.train_loader)

    def evaluate(self):
        self.generator.eval()
        total_loss = 0.0
        check_num = 10

        log_images = []
        log_mask_images = []
        log_edges = []
        log_mask_edges = []
        log_masks = []
        log_gen_images = []
        log_high_freq = []
        with torch.no_grad():
            cnt = 0
            for batch in tqdm(self.val_loader, desc="Evaluation"):
                g_loss, d_loss, losses, outputs = self.compute_losses(batch, train=False)
                total_loss += g_loss.item()

                log_images.append(batch['img'][0].cpu())
                log_mask_images.append(batch['mask_img'][0].cpu())
                log_edges.append(batch['edge'][0].cpu())
                log_mask_edges.append(batch['mask_edge_img'][0].cpu())
                log_masks.append(batch['mask'][0].cpu())
                log_gen_images.append(outputs['gen_img'][0].cpu())
                log_high_freq.append(outputs['high_freq'][0][0].cpu())
                cnt += 1
                if cnt == check_num:
                    break

        if self.use_wandb:
            wandb.log({
                "eval/g_loss": total_loss / check_num
            })
            wandb.log({
                "eval/images": [wandb.Image(img) for img in log_images],
                "eval/mask_images": [wandb.Image(mask_img) for mask_img in log_mask_images],
                "eval/edges": [wandb.Image(edge) for edge in log_edges],
                "eval/mask_edges": [wandb.Image(mask_edge) for mask_edge in log_mask_edges],
                "eval/masks": [wandb.Image(mask) for mask in log_masks],
                "eval/gen_images": [wandb.Image(gen_img) for gen_img in log_gen_images],
                "eval/high_freq": [wandb.Image(hf) for hf in log_high_freq]
            })

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
                    "epoch/time": time.time() - start_time
                })
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch + 1)
        torch.save(self.generator.state_dict(), os.path.join(self.checkpoint_dir, "final_generator.pt"))
        torch.save(self.discriminator.state_dict(), os.path.join(self.checkpoint_dir, "final_discriminator.pt"))
