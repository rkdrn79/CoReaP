from torch import nn
from typing import Dict, List, Tuple, Optional, Any, Union
from transformers.trainer import Trainer
from torch import nn
import torch
import numpy as np

class CoReaPTrainer(Trainer):
    def __init__(self, **kwds):
        super().__init__(**kwds)

        self.criteria = nn.CrossEntropyLoss()
        
    def compute_loss(self, model, inputs, return_outputs=False):
        mask_img = inputs['mask_img']
        mask = inputs['mask']
        mask_edge_img = inputs['mask_edge_img']
        mask_line_img = inputs['mask_line_img']
        label = inputs['img']
        
        output = model.forward(mask_img, mask, mask_edge_img, mask_line_img)

        loss = self.criteria(output, label)

        if return_outputs:
            return loss, output, label
        
        return loss

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)
        
        # Forward pass
        loss = self.compute_loss(model, inputs)
        
        # Backward pass (여기서 커스터마이징 가능)
        loss = loss / self.args.gradient_accumulation_steps  # Gradient Accumulation 고려
        loss.backward()
        
        # 예: gradient clipping 추가
        if self.args.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
        
        return loss.detach()
    
    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        model.eval()
        
        with torch.no_grad():
            eval_loss, pred, label = self.compute_loss(model,inputs,True)
        
        return (eval_loss,pred,label)

class CoReaPTrainer(Trainer):
    def __init__(self, generator, discriminator, g_optimizer, d_optimizer, **kwargs):
        """
        Custom Trainer for GAN.
        
        Args:
            generator: Generator model.
            discriminator: Discriminator model.
            g_optimizer: Optimizer for generator.
            d_optimizer: Optimizer for discriminator.
            g_loss_fn: Loss function for generator.
            d_loss_fn: Loss function for discriminator.
            **kwargs: Other arguments for Hugging Face Trainer.
        """
        super().__init__(**kwargs)
        self.generator = generator
        self.discriminator = discriminator
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Overwrite compute_loss to define GAN-specific loss computation.
        """
        mask_img = inputs['mask_img']
        mask = inputs['mask']
        mask_edge_img = inputs['mask_edge_img']
        mask_line_img = inputs['mask_line_img']
        label = inputs['img']

        batch_size = mask_img.size(0)

        # TODO: Add model forward pass and backward pass here

        # Combine losses for logging
        combined_loss = d_loss + g_loss

        return (combined_loss, {"d_loss": d_loss, "g_loss": g_loss}) if return_outputs else combined_loss

    def training_step(self, model, inputs):
        """
        Custom training step to handle the two models.
        """
        loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
        self.log({"train_loss": loss.item(), "d_loss": outputs["d_loss"].item(), "g_loss": outputs["g_loss"].item()})
        return loss
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Custom prediction step for GAN to compute loss during prediction.
        """
        real_data = inputs["real_data"]
        batch_size = real_data.size(0)

        self.generator.eval()
        self.discriminator.eval()

        # TODO: Add model forward pass here

        # If prediction_loss_only, return only loss
        if prediction_loss_only:
            return combined_loss, None, None

        # Otherwise, return loss and predictions
        return combined_loss, fake_data, real_data