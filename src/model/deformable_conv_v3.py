import torch
import torch.nn as nn
import numpy as np
from einops import rearrange

class Deformable_Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=2,
                 bias=None, modulation=True, adaptive_d=True):
        super(Deformable_Conv2d, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.adaptive_d = adaptive_d
        self.modulation = modulation
        
        # Zero padding layer
        self.zero_padding = nn.ZeroPad2d(padding)
        
        # Main convolution - stride is kernel_size to match your implementation
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                            stride=kernel_size, bias=bias)
        
        # Offset prediction conv
        self.p_conv = nn.Conv2d(in_channels, 2*kernel_size**2, kernel_size=3, 
                               padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        if self.p_conv.bias is not None:
            nn.init.constant_(self.p_conv.bias, 0)
        
        # Register backward hook for learning rate adjustment
        self.p_conv.register_full_backward_hook(self._set_lr)
        
        # Modulation conv if enabled
        if self.modulation:
            self.m_conv = nn.Conv2d(in_channels, kernel_size*kernel_size, 
                                  kernel_size=3, padding=1, stride=stride, bias=False)
            nn.init.constant_(self.m_conv.weight, 0.5)
            self.m_conv.register_full_backward_hook(self._set_lr)
        
        # Adaptive dilation conv if enabled
        if self.adaptive_d:
            self.ad_conv = nn.Conv2d(in_channels, kernel_size, kernel_size=3, 
                                   padding=1, stride=stride, bias=False)
            nn.init.constant_(self.ad_conv.weight, 1)
            self.ad_conv.register_full_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        # Adjust learning rate without affecting downstream gradients
        new_grad_input = tuple(grad * 0.1 for grad in grad_input)
        return new_grad_input

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2 + 1),
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2 + 1),
            indexing='ij'
        )
        p_n = torch.cat([p_n_x.flatten(), p_n_y.flatten()], 0)
        p_n = p_n.view(1, 2*N, 1, 1).type(dtype)
        
        return p_n

    def _get_p_0(self, N, h, w, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h * self.stride + 1, self.stride),
            torch.arange(1, w * self.stride + 1, self.stride),
            indexing='ij'
        )
        
        p_0_x = p_0_x.flatten().view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = p_0_y.flatten().view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)
        
        return p_0

    def _get_p(self, offset, dtype, ad_offset):
        N = offset.size(1) // 2
        p_n = self._get_p_n(N, dtype)
        p_0 = self._get_p_0(N, offset.size(2), offset.size(3), dtype)
        
        if self.adaptive_d and ad_offset is not None:
            p = p_0 + p_n + offset + ad_offset * p_n
        else:
            p = p_0 + p_n + offset
            
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        
        x = x.contiguous().view(b, c, -1)
        
        index = q[..., :N] * padded_w + q[..., N:]
        index = index.contiguous().unsqueeze(1).expand(-1, c, -1, -1, -1)
        index = index.contiguous().view(b, c, -1).long()
        
        x_offset = x.gather(dim=-1, index=index)
        x_offset = x_offset.view(b, c, h, w, N)
        
        return x_offset

    def _reshape_x_offset(self, x_offset, ks):
        b, c, h, w, n = x_offset.size()
        x_offset = rearrange(x_offset, 'b c h w (kh kw) -> b c (h kh) (w kw)', 
                           kh=ks, kw=ks)
        return x_offset

    def forward(self, x):
        # Get offsets
        offset = self.p_conv(x)
        
        # Store offset for mask forward
        self.offset = offset
        
        # Handle adaptive dilation
        if self.adaptive_d:
            ad_base = self.ad_conv(x)
            ad_base = 1 - torch.sigmoid(ad_base)
            ad = ad_base.repeat(1, 2 * self.kernel_size, 1, 1) * self.dilation
            
            ad_m = (ad_base - 0.5) * 2
            ad_m = ad_m.repeat(1, self.kernel_size, 1, 1) * self.dilation
            self.ad_m = ad_m.detach()
        
        # Apply padding if needed
        if self.padding:
            x = self.zero_padding(x)
        
        # Get sampling locations
        if self.adaptive_d:
            p = self._get_p(offset, offset.dtype, ad)
        else:
            p = self._get_p(offset, offset.dtype, None)
            
        p = p.contiguous().permute(0, 2, 3, 1)
        
        # Create mask for points outside the image boundary
        N = offset.size(1) // 2
        mask = torch.cat([
            p[..., :N].lt(self.padding) + p[..., :N].gt(x.size(2) - 1 - self.padding),
            p[..., N:].lt(self.padding) + p[..., N:].gt(x.size(3) - 1 - self.padding)
        ], dim=-1).type_as(p)
        mask = mask.detach()
        
        # Store mask for later use
        self.mask = mask
        
        # Apply mask to sampling points
        floor_p = torch.floor(p)
        p = p * (1 - mask) + floor_p * mask
        p = torch.cat([
            torch.clamp(p[..., :N], 0, x.size(2) - 1),
            torch.clamp(p[..., N:], 0, x.size(3) - 1)
        ], dim=-1)
        
        # Store processed p for mask forward
        self.p = p.detach()
        
        # Calculate bilinear sampling coordinates
        q_lt = p.detach().floor()
        q_rb = q_lt + 1
        
        q_lt = torch.cat([
            torch.clamp(q_lt[..., :N], 0, x.size(2)-1),
            torch.clamp(q_lt[..., N:], 0, x.size(3)-1)
        ], dim=-1).long()
        
        q_rb = torch.cat([
            torch.clamp(q_rb[..., :N], 0, x.size(2)-1),
            torch.clamp(q_rb[..., N:], 0, x.size(3)-1)
        ], dim=-1).long()
        
        # Store q points for mask forward
        self.q_lt = q_lt
        self.q_rb = q_rb
        
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], -1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], -1)
        
        # Store remaining q points
        self.q_lb = q_lb
        self.q_rt = q_rt
        
        # Get pixel values
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)
        
        # Calculate bilinear weights
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * \
               (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * \
               (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * \
               (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * \
               (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))
        
        # Apply bilinear interpolation
        x_offset = g_lt.unsqueeze(1) * x_q_lt + \
                  g_rb.unsqueeze(1) * x_q_rb + \
                  g_lb.unsqueeze(1) * x_q_lb + \
                  g_rt.unsqueeze(1) * x_q_rt
        
        # Apply modulation if enabled
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))
            
            if self.adaptive_d:
                m = m * ad_m
                
            m = m.contiguous().permute(0, 2, 3, 1)
            m = m.unsqueeze(1)
            self.m = m.detach()
            
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m
        
        # Reshape and apply final convolution
        x_offset = self._reshape_x_offset(x_offset, self.kernel_size)
        out = self.conv(x_offset)
        
        return out

    def _forward_mask(self, x):
        """
        Forward pass for mask visualization
        """
        if self.padding:
            x = self.zero_padding(x)
        
        N = self.offset.size(1) // 2
        
        # Calculate bilinear weights
        g_lt = (1 + (self.q_lt[..., :N].type_as(self.p) - self.p[..., :N])) * \
               (1 + (self.q_lt[..., N:].type_as(self.p) - self.p[..., N:]))
        g_rb = (1 - (self.q_rb[..., :N].type_as(self.p) - self.p[..., :N])) * \
               (1 - (self.q_rb[..., N:].type_as(self.p) - self.p[..., N:]))
        g_lb = (1 + (self.q_lb[..., :N].type_as(self.p) - self.p[..., :N])) * \
               (1 - (self.q_lb[..., N:].type_as(self.p) - self.p[..., N:]))
        g_rt = (1 - (self.q_rt[..., :N].type_as(self.p) - self.p[..., :N])) * \
               (1 + (self.q_rt[..., N:].type_as(self.p) - self.p[..., N:]))
        
        # Get pixel values
        x_q_lt = self._get_x_q(x, self.q_lt, N)
        x_q_rb = self._get_x_q(x, self.q_rb, N)
        x_q_lb = self._get_x_q(x, self.q_lb, N)
        x_q_rt = self._get_x_q(x, self.q_rt, N)
        
        # Apply bilinear interpolation
        x_offset = g_lt.unsqueeze(1) * x_q_lt + \
                  g_rb.unsqueeze(1) * x_q_rb + \
                  g_lb.unsqueeze(1) * x_q_lb + \
                  g_rt.unsqueeze(1) * x_q_rt
        
        if self.modulation:
            m = torch.cat([self.m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m
        
        x_offset = self._reshape_x_offset(x_offset, self.kernel_size)
        out = self.conv(x_offset)
        
        return out