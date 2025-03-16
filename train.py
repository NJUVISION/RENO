import os
import random
import argparse
import datetime

import numpy as np
from glob import glob

import torch
import torch.utils.data
from torch import nn
from torch.cuda import amp

from torchsparse.nn import functional as F
from torchsparse.utils.collate import sparse_collate_fn

from dataset import PCDataset
from network import Network

seed = 11
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

device='cuda:0'

# set torchsparse config
conv_config = F.conv_config.get_default_conv_config()
conv_config.kmap_mode = "hashmap"
F.conv_config.set_global_conv_config(conv_config)

parser = argparse.ArgumentParser(
    prog='train.py',
    description='Training from scratch.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('--training_data', default='/root/Dataset/KITTI_detection/training/velodyne/*.ply', help='Training data (Glob pattern).')
parser.add_argument('--model_save_folder', default='./model/KITTIDetection', help='Directory where to save trained models.')
parser.add_argument("--is_data_pre_quantized", type=bool, default=False, help="Whether the training data is pre quantized.")
parser.add_argument("--valid_samples", type=str, default='', help="Something like train.txt/val.txt.")

parser.add_argument('--channels', type=int, help='Neural network channels.', default=32)
parser.add_argument('--kernel_size', type=int, help='Convolution kernel size.', default=3)

parser.add_argument('--batch_size', type=int, help='Batch size.', default=1)
parser.add_argument('--learning_rate', type=float, help='Learning rate.', default=0.0005)
parser.add_argument('--lr_decay', type=float, help='Decays the learning rate to x times the original.', default=0.1)
parser.add_argument('--lr_decay_steps', help='Decays the learning rate at x steps.', default=[100000, 150000])
parser.add_argument('--max_steps', type=int, help='Train up to this number of steps.', default=170000)

args = parser.parse_args()

# CREATE MODEL SAVE PATH
os.makedirs(args.model_save_folder, exist_ok=True)

files = np.array(glob(args.training_data, recursive=True))

# check training samples
if args.valid_samples != '':
    valid_sample_names = np.loadtxt(args.valid_samples, dtype=str)
    valid_files = []
    for f in files:
        fname = f.split('/')[-1].split('.')[0]
        if fname in valid_sample_names:
            valid_files.append(f)
    files = valid_files

np.random.shuffle(files)
files = files[:]

dataflow = torch.utils.data.DataLoader(
    dataset=PCDataset(files, is_pre_quantized=args.is_data_pre_quantized),
    shuffle=True,
    batch_size=args.batch_size,
    collate_fn=sparse_collate_fn
)

net = Network(channels=args.channels, kernel_size=args.kernel_size).to(device).train()
optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)

losses = []
global_step = 0

for epoch in range(1, 9999):
    print(datetime.datetime.now())
    for data in dataflow:
        x = data['input'].to(device=device)
        
        optimizer.zero_grad()
        loss = net(x)

        loss.backward()
        optimizer.step()
        global_step += 1

        # PRINT
        losses.append(loss.item())

        if global_step % 500 == 0:
            print(f'Epoch:{epoch} | Step:{global_step} | Loss:{round(np.array(losses).mean(), 5)}')
            losses = []

         # LEARNING RATE DECAY
        if global_step in args.lr_decay_steps:
            args.learning_rate = args.learning_rate * args.lr_decay
            for g in optimizer.param_groups:
                g['lr'] = args.learning_rate
            print(f'Learning rate decay triggered at step {global_step}, LR is setting to {args.learning_rate}.')

        # SAVE MODEL
        if global_step % 500 == 0:
            torch.save(net.state_dict(), os.path.join(args.model_save_folder, 'ckpt.pt'))
        
        if global_step >= args.max_steps:
            break

    if global_step >= args.max_steps:
        break
