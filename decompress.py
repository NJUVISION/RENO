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
    prog='decompress.py',
    description='Decompress point cloud geometry.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('--input_glob', default='./data/kittdet_compressed/*.bin', help='Glob pattern for input bin files.')
parser.add_argument('--output_folder', default='./data/kittdet_decompressed/', help='Folder to save decompressed ply files.')
parser.add_argument("--is_data_pre_quantized", type=bool, default=False, help="Whether the original point cloud is pre quantized.")

parser.add_argument('--channels', type=int, help='Neural network channels.', default=32)
parser.add_argument('--kernel_size', type=int, help='Convolution kernel size.', default=3)
parser.add_argument('--ckpt', help='Checkpoint load path.', default='./model/KITTIDetection/ckpt.pt')

args = parser.parse_args()

os.makedirs(args.output_folder, exist_ok=True)

file_path_ls = glob(args.input_glob)

# network
net = Network(channels=args.channels, kernel_size=args.kernel_size)
net.load_state_dict(torch.load(args.ckpt))
net.cuda().eval()

# warm up
random_coords = torch.randint(low=0, high=4096, size=(4096, 3)).int().cuda()
net(SparseTensor(coords=torch.cat((random_coords[:, 0:1]*0, random_coords), dim=-1),
                feats=torch.ones((4096, 1))).cuda())

dec_time_ls = []

with torch.no_grad():
    for file_path in tqdm(file_path_ls):
        file_name = os.path.split(file_path)[-1]
        decompressed_file_path = os.path.join(args.output_folder, file_name+'.ply')

        ################################ Read bin file

        with open(file_path, 'rb') as f:
            posQ = np.frombuffer(f.read(2), dtype=np.float16)[0]
            base_x_len = np.frombuffer(f.read(4), dtype=np.int32)[0]
            base_x_coords = np.frombuffer(f.read(base_x_len*4*3), dtype=np.int32)
            base_x_feats = np.frombuffer(f.read(base_x_len*1), dtype=np.uint8)
            byte_stream = f.read()

        dec_time_start = time.time()

        ################################ Decompress

        base_x_coords = torch.tensor(base_x_coords.reshape(-1, 3), device=device) 
        base_x_feats = torch.tensor(base_x_feats.reshape(-1, 1), device=device)

        x = SparseTensor(coords=torch.cat((base_x_feats*0, base_x_coords), dim=-1), feats=base_x_feats).cuda()
        byte_stream_ls = op.unpack_byte_stream(byte_stream)

        for byte_stream_idx in range(0, len(byte_stream_ls), 2):
            byte_stream_s0 = byte_stream_ls[byte_stream_idx]
            byte_stream_s1 = byte_stream_ls[byte_stream_idx+1]

            # embedding prior scale feats
            x_O = x.feats.int()
            x.feats = net.prior_embedding(x_O).view(-1, net.channels) # (N_d, C)
            x = net.prior_resnet(x) # (N_d, C)

            # target embedding
            x_up_C, x_up_F = net.fcg(x.coords, x_O, x_F=x.feats)
            x_up_C, x_up_F = op.sort_CF(x_up_C, x_up_F)

            x_up_F = net.target_embedding(x_up_F, x_up_C)
            x_up = SparseTensor(coords=x_up_C, feats=x_up_F)
            x_up = net.target_resnet(x_up)


            x_up_O_prob_s0 = net.pred_head_s0(x_up.feats) # (B*Nt, 16)
            x_up_O_cdf_s0 = torch.cat((x_up_O_prob_s0[:, 0:1]*0, x_up_O_prob_s0.cumsum(dim=-1)), dim=-1) # (B*Nt, 16)
            x_up_O_cdf_s0 = torch.clamp(x_up_O_cdf_s0, min=0, max=1)
            x_up_O_cdf_s0_norm = op._convert_to_int_and_normalize(x_up_O_cdf_s0, True)
            x_up_O_cdf_s0_norm = x_up_O_cdf_s0_norm.cpu()
            x_up_O_s0 = torchac.decode_int16_normalized_cdf(x_up_O_cdf_s0_norm, byte_stream_s0).cuda()

            x_up_O_prob_s1 = net.pred_head_s1(x_up.feats + net.pred_head_s1_emb(x_up_O_s0.long())) # (B*Nt, 16)
            x_up_O_cdf_s1 = torch.cat((x_up_O_prob_s1[:, 0:1]*0, x_up_O_prob_s1.cumsum(dim=-1)), dim=-1) # (B*Nt, 16)
            x_up_O_cdf_s1 = torch.clamp(x_up_O_cdf_s1, min=0, max=1)
            x_up_O_cdf_s1_norm = op._convert_to_int_and_normalize(x_up_O_cdf_s1, True)
            x_up_O_cdf_s1_norm = x_up_O_cdf_s1_norm.cpu()
            x_up_O_s1 = torchac.decode_int16_normalized_cdf(x_up_O_cdf_s1_norm, byte_stream_s1).cuda()

            x_up_O = x_up_O_s1 * 16 + x_up_O_s0

            x = SparseTensor(coords=x_up_C, feats=x_up_O.unsqueeze(-1)).cuda()
            
        # decode the last layer
        scan = net.fcg(x.C, x.F)

        if args.is_data_pre_quantized:
            scan = scan[:, 1:] * posQ
        else:
            scan = (scan[:, 1:] * posQ - 131072) * 0.001

        dec_time_end = time.time()

        dec_time_ls.append(dec_time_end-dec_time_start)

        io.save_ply_ascii_geo(scan.float().cpu().numpy(), decompressed_file_path)

print('Total: {total_n:d} | Decode Time:{dec_time:.3f} | Max GPU Memory:{memory:.2f}MB'.format(
    total_n=len(dec_time_ls),
    dec_time=np.array(dec_time_ls).mean(),
    memory=torch.cuda.max_memory_allocated()/1024/1024
))
