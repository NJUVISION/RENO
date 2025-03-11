import torch
from torchsparse import SparseTensor
import kit.io as io

class PCDataset:
    def __init__(self, file_path_ls, posQ=4, is_pre_quantized=False):
        self.files = io.read_point_clouds(file_path_ls)
        self.posQ = posQ
        self.is_pre_quantized = is_pre_quantized

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        xyz = torch.tensor(self.files[idx], dtype=torch.float)
        feats = torch.ones((xyz.shape[0], 1), dtype=torch.float)

        if not self.is_pre_quantized:
            xyz = xyz / 0.001 
        xyz = torch.round((xyz + 131072) / self.posQ).int()

        input = SparseTensor(coords=xyz, feats=feats)
        
        return {"input": input}
