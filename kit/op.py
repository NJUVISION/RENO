import os
import torch
import numpy as np


def sort_C(C):
    _, indices = torch.sort(C[:, 1])
    C = C[indices]
    _, indices = torch.sort(C[:, 2], stable=True)
    C = C[indices]
    _, indices = torch.sort(C[:, 3], stable=True)
    C = C[indices]
    _, indices = torch.sort(C[:, 0], stable=True)
    C = C[indices]
    return C

def sort_CF(C, F):
    _, indices = torch.sort(C[:, 1])
    C = C[indices]
    F = F[indices]
    _, indices = torch.sort(C[:, 2], stable=True)
    C = C[indices]
    F = F[indices]
    _, indices = torch.sort(C[:, 3], stable=True)
    C = C[indices]
    F = F[indices]
    _, indices = torch.sort(C[:, 0], stable=True)
    C = C[indices]
    F = F[indices]
    return C, F

def pack_byte_stream_ls(byte_stream_ls):
    stream = np.array(len(byte_stream_ls), dtype=np.uint16).tobytes()
    for byte_stream in byte_stream_ls:
        stream += np.array(len(byte_stream), dtype=np.uint32).tobytes()
        stream += byte_stream
    return stream

def unpack_byte_stream(stream):
    len_bytes_stream_ls = np.frombuffer(stream[:2], dtype=np.uint16)[0]
    byte_stream_ls = []
    cursor = 2
    for idx in range(len_bytes_stream_ls):
        len_bytes_stream = np.frombuffer(stream[cursor:cursor+4], dtype=np.uint32)[0]
        byte_stream = stream[cursor+4:cursor+4+len_bytes_stream]
        byte_stream_ls.append(byte_stream)
        cursor = cursor + 4 + len_bytes_stream
    return byte_stream_ls

def _convert_to_int_and_normalize(cdf_float, needs_normalization):
  """Convert floatingpoint CDF to integers. See README for more info.

  The idea is the following:
  When we get the cdf here, it is (assumed to be) between 0 and 1, i.e,
    cdf \in [0, 1)
  (note that 1 should not be included.)
  We now want to convert this to int16 but make sure we do not get
  the same value twice, as this would break the arithmetic coder
  (you need a strictly monotonically increasing function).
  So, if needs_normalization==True, we multiply the input CDF
  with 2**16 - (Lp - 1). This means that now,
    cdf \in [0, 2**16 - (Lp - 1)].
  Then, in a final step, we add an arange(Lp), which is just a line with
  slope one. This ensure that for sure, we will get unique, strictly
  monotonically increasing CDFs, which are \in [0, 2**16)
  """
  Lp = cdf_float.shape[-1]
  factor = torch.tensor(2, dtype=torch.float32, device=cdf_float.device).pow_(16)
  new_max_value = factor
  if needs_normalization:
    new_max_value = new_max_value - (Lp - 1)
  cdf_float = cdf_float.mul(new_max_value)
  cdf_float = cdf_float.round()

  cdf = cdf_float.to(dtype=torch.int16, non_blocking=True)
  if needs_normalization:
    r = torch.arange(Lp, dtype=torch.int16, device=cdf.device)
    cdf.add_(r)
  return cdf

def get_file_size_in_bits(file_path):
    return os.stat(file_path).st_size * 8