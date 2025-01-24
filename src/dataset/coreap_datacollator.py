import torch
import numpy as np

class CoReaPDataCollator:
    def __init__(self, args):
        self.args = args

    def __call__(self, features):

        mask_img = [torch.tensor(feature['mask_img']) for feature in features]
        mask_img = torch.stack(mask_img).to(torch.bfloat16 if self.args.bf16 else torch.float16)
        
        mask = [torch.tensor(feature['mask']) for feature in features]
        mask = torch.stack(mask).to(torch.bfloat16 if self.args.bf16 else torch.float16)

        mask_edge_img = [torch.tensor(feature['mask_edge_img']) for feature in features]
        mask_edge_img = torch.stack(mask_edge_img).to(torch.bfloat16 if self.args.bf16 else torch.float16)

        mask_line_img = [torch.tensor(feature['mask_line_img']) for feature in features]
        mask_line_img = torch.stack(mask_line_img).to(torch.bfloat16 if self.args.bf16 else torch.float16)
                                                      
        img = [torch.tensor(feature['img']) for feature in features]
        img = torch.stack(img).to(torch.bfloat16 if self.args.bf16 else torch.float16)

        return {
            'mask_img': mask_img,
            'mask': mask,
            'mask_edge_img': mask_edge_img,
            'mask_line_img': mask_line_img,
            'img': img
        }