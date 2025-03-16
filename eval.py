import os
import argparse
import subprocess

import numpy as np
from glob import glob
from tqdm import tqdm

from multiprocessing import Pool

parser = argparse.ArgumentParser(
    prog='eval.py',
    description='Eval geometry PSNR.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('--input_glob', type=str, help='Glob pattern to load point clouds.', default='./data/examples/*.ply')
parser.add_argument('--decompressed_path', type=str, help='Path to save decompressed files.', default='./data/kitt_decompressed')
parser.add_argument('--pcc_metric_path', type=str, help='Path for pc_error_d.', default='./third_party/pc_error_d')
parser.add_argument('--resolution', type=float, help='Point cloud resolution (peak signal).', default=59.70)

args = parser.parse_args()

files = np.array(glob(args.input_glob))

# check if decode file exists
checked_files = []
for file in files:
    filename_wo_ext = os.path.split(file)[-1].split('.ply')[0]
    dec_f = os.path.join(os.path.abspath(args.decompressed_path), filename_wo_ext+'.ply.bin.ply')
    if os.path.exists(dec_f):
        checked_files.append(file)
files = checked_files

def process(input_f):
    filename_wo_ext = os.path.split(input_f)[-1].split('.ply')[0]
    dec_f = os.path.join(os.path.abspath(args.decompressed_path), filename_wo_ext+'.ply.bin.ply')
    cmd = f'{args.pcc_metric_path} \
    --fileA={input_f} --fileB={dec_f} \
    --resolution={args.resolution} --inputNorm={input_f}'
    try:
        output = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
        d1_psnr = float(str(output).split('mseF,PSNR (p2point):')[1].split('\\n')[0])
        d2_psnr = float(str(output).split('mseF,PSNR (p2plane):')[1].split('\\n')[0])
    except:
        print('!!!Error!!!', cmd)
        d1_psnr, d2_psnr = -1, -1
        
    return np.array([filename_wo_ext, d1_psnr, d2_psnr])


with Pool(32) as p:
    arr = list(tqdm(p.imap(process, files), total=len(files)))

# process(files[0])

arr = np.array(arr)
fnames, d1_PSNRs, d2_PSNRs = arr[:, 0], arr[:, 1].astype(float), arr[:, 2].astype(float)

print('Avg. D1 PSNR:', round(d1_PSNRs.mean(), 3))
print('Avg. D2 PSNR:', round(d2_PSNRs.mean(), 3))
