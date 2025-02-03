from torch import nn
from typing import Dict, List, Tuple, Optional, Any, Union
from transformers.trainer import Trainer
from torch import nn
import torch
import numpy as np
from src.utils.losses.pcp import PerceptualLoss

class CoReaPTrainer(Trainer):
    def __init__(self, generator, discriminator, g_optimizer, d_optimizer, 
                r1_gamma=10, pcp_ratio=1.0, l1_ratio=1.0, **kwargs):
        super().__init__(**kwargs)
        self.generator = generator
        self.discriminator = discriminator
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.r1_gamma = r1_gamma
        self.pcp_ratio = pcp_ratio
        self.l1_ratio = l1_ratio
        self.pcp = PerceptualLoss(layer_weights=dict(conv4_4=1/4, conv5_4=1/2)).to(self.args.device)

    def compute_loss(self, model, inputs, return_outputs=False):
        # 데이터 준비
        mask_img = inputs['mask_img'].to(self.args.device)
        mask = inputs['mask'].to(self.args.device)
        label = inputs['img'].to(self.args.device)
        
        # Generator 포워드 패스
        gen_img, gen_img_stg1 = self.generator(mask_img, mask)
        
        # Discriminator 포워드 패스
        with torch.no_grad():
            real_logits, real_logits_stg1 = self.discriminator(label, mask, label)
            fake_logits, fake_logits_stg1 = self.discriminator(gen_img.detach(), mask, gen_img_stg1.detach())
        
        # Generator Loss 계산
        g_loss_gan = torch.nn.functional.softplus(-fake_logits).mean()
        g_loss_l1 = torch.nn.functional.l1_loss(gen_img, label)
        pcp_loss, _ = self.pcp(gen_img, label)
        g_loss = g_loss_gan + self.pcp_ratio * pcp_loss + self.l1_ratio * g_loss_l1
        
        # Discriminator Loss 계산
        d_loss_real = torch.nn.functional.softplus(-real_logits).mean()
        d_loss_fake = torch.nn.functional.softplus(fake_logits).mean()
        d_loss = d_loss_real + d_loss_fake
        
        # R1 Regularization
        if self.r1_gamma > 0:
            real_img_tmp = label.detach().requires_grad_(True)
            real_logits_tmp, _ = self.discriminator(real_img_tmp, mask, real_img_tmp)
            r1_grads = torch.autograd.grad(
                outputs=real_logits_tmp.sum(), 
                inputs=real_img_tmp, 
                create_graph=True
            )[0]
            r1_penalty = r1_grads.square().sum([1,2,3]).mean()
            d_loss += self.r1_gamma * 0.5 * r1_penalty

        combined_loss = g_loss + d_loss
        
        if return_outputs:
            return (combined_loss, {
                "g_loss": g_loss,
                "d_loss": d_loss,
                "gen_img": gen_img,
                "real_img": label
            })
        return combined_loss

    def training_step(self, model, inputs):
        # Discriminator 업데이트
        self.discriminator.requires_grad_(True)
        self.generator.requires_grad_(False)
        
        loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
        self.d_optimizer.zero_grad()
        outputs['d_loss'].backward(retain_graph=True)
        self.d_optimizer.step()
        
        # Generator 업데이트
        self.discriminator.requires_grad_(False)
        self.generator.requires_grad_(True)
        
        self.g_optimizer.zero_grad()
        outputs['g_loss'].backward()
        self.g_optimizer.step()
        
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only=False, ignore_keys=None):
        # Validation을 위한 예측 단계
        self.generator.eval()
        self.discriminator.eval()
        
        with torch.no_grad():
            mask_img = inputs['mask_img'].to(self.args.device)
            mask = inputs['mask'].to(self.args.device)
            label = inputs['img'].to(self.args.device)
            
            gen_img, _ = self.generator(mask_img, mask)
            loss = self.compute_loss(model, inputs)
            
        if prediction_loss_only:
            return loss.detach(), None, None
            
        return loss.detach(), gen_img, label