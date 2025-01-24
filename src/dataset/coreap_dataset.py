import torch
import numpy as np

from torch.utils.data import Dataset

from src.dataset.preprocessor import *
# preprocessor classes are imported <- all preprocessor classes are imported

class CoReaPDataset(Dataset):
    def __init__(self, args, data):
        self.args = args
        self.data = data

    def __len__(self):
        return len(self.data['mask_img'])
    å
    def __getitem__(self, idx):

        mask_img = self.data['mask_img'][idx]
        mask = self.data['mask'][idx]

        mask_edge_img = self.data['mask_edge_img'][idx]
        mask_line_img = self.data['mask_line_img'][idx]

        img = self.data['img'][idx]
        return {
            'mask_img': mask_img,
            'mask': mask,
            'mask_edge_img': mask_edge_img,
            'mask_line_img': mask_line_img,
            'img': img
        }
    