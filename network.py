import torch
import torch.nn as nn

import torchsparse
from torchsparse import nn as spnn
from torchsparse import SparseTensor

import kit.op as op
from kit.nn import ResNet, FOG, FCG, TargetEmbedding

class Network(nn.Module):
    def __init__(self, channels, kernel_size):
        super(Network, self).__init__()

        self.prior_embedding = nn.Embedding(256, channels)

        self.prior_resnet = nn.Sequential(
            spnn.Conv3d(channels, channels, kernel_size),
            spnn.ReLU(True),
            ResNet(channels, k=kernel_size),
            ResNet(channels, k=kernel_size),
        )
        
        ###########################

        self.target_embedding = TargetEmbedding(channels)

        self.target_resnet = nn.Sequential(
            spnn.Conv3d(channels, channels, kernel_size),
            spnn.ReLU(True),
            ResNet(channels, k=kernel_size),
            ResNet(channels, k=kernel_size),
        )

        ###########################

        self.pred_head_s0 = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(True),
            nn.Linear(channels, 16),
            nn.Softmax(dim=-1),
        )

        self.pred_head_s1_emb = nn.Embedding(16, channels)
        self.pred_head_s1 = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(True),
            nn.Linear(channels, 16),
            nn.Softmax(dim=-1),
        )

        self.channels = channels
        self.fog = FOG()
        self.fcg = FCG()

    def forward(self, x):
        N = x.coords.shape[0]

        # get sparse occupancy code list
        data_ls = []
        while True:
            x = self.fog(x)
            data_ls.append((x.coords.clone(), x.feats.clone())) # must clone
            if x.coords.shape[0] < 64:
                break
        data_ls = data_ls[::-1]
        # data_ls: [(coords, occupancy), (coords, occupancy), ...]

        total_bits = 0
        
        for depth in range(len(data_ls)-1):
            x_C, x_O = data_ls[depth]
            gt_x_up_C, gt_x_up_O = data_ls[depth+1]
            gt_x_up_C, gt_x_up_O = op.sort_CF(gt_x_up_C, gt_x_up_O)

            # embedding prior scale feats
            x_F = self.prior_embedding(x_O.int()).view(-1, self.channels) # (N_d, C)
            x = SparseTensor(coords=x_C, feats=x_F)
            x = self.prior_resnet(x) # (N_d, C) 

            # target embedding
            x_up_C, x_up_F = self.fcg(x_C, x_O, x.feats)
            x_up_C, x_up_F = op.sort_CF(x_up_C, x_up_F)

            x_up_F = self.target_embedding(x_up_F, x_up_C)
            x_up = SparseTensor(coords=x_up_C, feats=x_up_F)
            x_up = self.target_resnet(x_up)

            # bit-wise two-stage coding
            gt_x_up_O_s0 = torch.remainder(gt_x_up_O, 16) # 8-4-2-1
            gt_x_up_O_s1 = torch.div(gt_x_up_O, 16, rounding_mode='floor') # 128-64-32-16

            x_up_O_prob_s0 = self.pred_head_s0(x_up.feats) # (B*Nt, 256)
            x_up_O_prob_s1 = self.pred_head_s1(x_up.feats + self.pred_head_s1_emb(gt_x_up_O_s0[:, 0].long())) # (B*Nt, 256)

            x_up_O_prob_s0 = x_up_O_prob_s0.gather(1, gt_x_up_O_s0.long()) # (B*Nt, 1)
            x_up_O_prob_s1 = x_up_O_prob_s1.gather(1, gt_x_up_O_s1.long()) # (B*Nt, 1)

            total_bits += torch.sum(torch.clamp(-1.0 * torch.log2(x_up_O_prob_s0 + 1e-10), 0, 50))
            total_bits += torch.sum(torch.clamp(-1.0 * torch.log2(x_up_O_prob_s1 + 1e-10), 0, 50))
            
        bpp = total_bits / N

        return bpp
