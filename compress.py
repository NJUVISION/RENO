import os
import time
import random
import argparse

import numpy as np
from glob import glob
from tqdm import tqdm

import torch
import torchac

from torchsparse import SparseTensor
from torchsparse.nn import functional as F

from network import Network

import kit.io as io
import kit.op as op

random.seed(1)
np.random.seed(1)
device = 'cuda'

# set torchsparse config
conv_config = F.conv_config.get_default_conv_config()
conv_config.kmap_mode = "hashmap"
F.conv_config.set_global_conv_config(conv_config)

parser = argparse.ArgumentParser(
    prog='compress.py',
    description='Compress point cloud geometry.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('--input_glob', default='./data/kittidet_examples/*.ply', help='Glob pattern for input point clouds.')
parser.add_argument('--output_folder', default='./data/kittidet_compressed/', help='Folder to save compressed bin files.')
parser.add_argument("--is_data_pre_quantized", type=bool, default=False, help="Whether the input data is pre quantized.")
parser.add_argument('--posQ', default=16, type=int, help='Quantization scale.')

parser.add_argument('--channels', type=int, help='Neural network channels.', default=32)
parser.add_argument('--kernel_size', type=int, help='Convolution kernel size.', default=3)
parser.add_argument('--ckpt', help='Checkpoint load path.', default='./model/KITTIDetection/ckpt.pt')

parser.add_argument('--num_samples', default=-1, help='Random choose some samples for quick test. [-1 means test all data]')

args = parser.parse_args()

os.makedirs(args.output_folder, exist_ok=True)

file_path_ls = glob(args.input_glob, recursive=True)

# random choose some samples for quick test
if args.num_samples != -1:
    np.random.shuffle(file_path_ls)
    file_path_ls = file_path_ls[:args.num_samples]

# reading point cloud using multithread
xyz_ls = io.read_point_clouds(file_path_ls)

# network
net = Network(channels=args.channels, kernel_size=args.kernel_size)
net.load_state_dict(torch.load(args.ckpt))
net.cuda().eval()

# warm up
random_coords = torch.randint(low=0, high=2048, size=(2048, 3)).int().cuda()
net(SparseTensor(coords=torch.cat((random_coords[:, 0:1]*0, random_coords), dim=-1),
                feats=torch.ones((2048, 1))).cuda())

enc_time_ls, bpp_ls = [], []

with torch.no_grad():
    for file_idx in tqdm(range(len(file_path_ls))):
        file_path = file_path_ls[file_idx]
        file_name = os.path.split(file_path)[-1]
        compressed_file_path = os.path.join(args.output_folder, file_name+'.bin')

        ################################ Get xyz

        if args.is_data_pre_quantized:
            xyz = torch.tensor(xyz_ls[file_idx])
        else:
            xyz = torch.tensor(xyz_ls[file_idx] / 0.001 + 131072)

        xyz = torch.round(xyz / args.posQ).int()
        N = xyz.shape[0]

        xyz = torch.cat((xyz[:,0:1]*0, xyz), dim=-1).int()
        feats = torch.ones((xyz.shape[0], 1), dtype=torch.float)
        x = SparseTensor(coords=xyz, feats=feats).cuda()

        torch.cuda.synchronize()
        enc_time_start = time.time()

        ################################ Preprocessing

        data_ls = []
        while True:
            x = net.fog(x)
            data_ls.append((x.coords.clone(), x.feats.clone())) # must clone
            if x.coords.shape[0] < 64:
                break
        data_ls = data_ls[::-1]

        ################################ NN Inference

        byte_stream_ls = []
        for depth in range(len(data_ls)-1):
            x_C, x_O = data_ls[depth]
            gt_x_up_C, gt_x_up_O = data_ls[depth+1]
            gt_x_up_C, gt_x_up_O = op.sort_CF(gt_x_up_C, gt_x_up_O)

            # embedding prior scale feats
            x_F = net.prior_embedding(x_O.int()).view(-1, net.channels) # (N_d, C)
            x = SparseTensor(coords=x_C, feats=x_F)
            x = net.prior_resnet(x) # (N_d, C) 

            # target embedding
            x_up_C, x_up_F = net.fcg(x_C, x_O, x.feats)
            x_up_C, x_up_F = op.sort_CF(x_up_C, x_up_F)

            x_up_F = net.target_embedding(x_up_F, x_up_C)
            x_up = SparseTensor(coords=x_up_C, feats=x_up_F)
            x_up = net.target_resnet(x_up)

            # bit-wise two-stage coding
            gt_x_up_O_s0 = torch.remainder(gt_x_up_O, 16) # 8-4-2-1
            gt_x_up_O_s1 = torch.div(gt_x_up_O, 16, rounding_mode='floor') # 128-64-32-16

            x_up_O_prob_s0 = net.pred_head_s0(x_up.feats) # (B*Nt, 256)
            x_up_O_prob_s1 = net.pred_head_s1(x_up.feats + net.pred_head_s1_emb(gt_x_up_O_s0[:, 0].long())) # (B*Nt, 256)

            # AE
            x_up_O_prob = torch.cat((x_up_O_prob_s0, x_up_O_prob_s1), dim=0)
            gt_x_up_O = torch.cat((gt_x_up_O_s0, gt_x_up_O_s1), dim=0)

            # get cdf
            x_up_O_cdf = torch.cat((x_up_O_prob[:, 0:1]*0, x_up_O_prob.cumsum(dim=-1)), dim=-1) # (Nt, 257)
            x_up_O_cdf = torch.clamp(x_up_O_cdf, min=0, max=1)
            x_up_O_cdf_norm = op._convert_to_int_and_normalize(x_up_O_cdf, True)

            # cdf to cpu
            x_up_O_cdf_norm = x_up_O_cdf_norm.cpu()
            gt_x_up_O = gt_x_up_O[:, 0].to(torch.int16).cpu()

            # coding
            half_num_gt_occ = gt_x_up_O.shape[0] // 2
            byte_stream_s0 = torchac.encode_int16_normalized_cdf(x_up_O_cdf_norm[:half_num_gt_occ], 
                                                              gt_x_up_O[:half_num_gt_occ])
            byte_stream_s1 = torchac.encode_int16_normalized_cdf(x_up_O_cdf_norm[half_num_gt_occ:], 
                                                              gt_x_up_O[half_num_gt_occ:])
            byte_stream_ls.append(byte_stream_s0)
            byte_stream_ls.append(byte_stream_s1)

        byte_stream = op.pack_byte_stream_ls(byte_stream_ls)

        torch.cuda.synchronize()
        enc_time_end = time.time()

        base_x_coords, base_x_feats = data_ls[0]
        base_x_len = base_x_coords.shape[0] 
        base_x_coords = base_x_coords[:, 1:].cpu().numpy() # (n, 3)
        base_x_feats = base_x_feats.cpu().numpy() # (n, 1)

        with open(compressed_file_path, 'wb') as f:
            f.write(np.array(args.posQ, dtype=np.float16).tobytes())
            f.write(np.array(base_x_len, dtype=np.int32).tobytes())
            f.write(np.array(base_x_coords, dtype=np.int32).tobytes())
            f.write(np.array(base_x_feats, dtype=np.uint8).tobytes())
            f.write(byte_stream)
        
        enc_time_ls.append(enc_time_end-enc_time_start)
        bpp_ls.append(op.get_file_size_in_bits(compressed_file_path)/N)

print('Total: {total_n:d} | Avg. Bpp:{bpp:.3f} | Encode time:{enc_time:.3f} | Max GPU Memory:{memory:.2f}MB'.format(
    total_n=len(enc_time_ls),
    bpp=np.array(bpp_ls).mean(),
    enc_time=np.array(enc_time_ls).mean(),
    memory=torch.cuda.max_memory_allocated()/1024/1024
))

