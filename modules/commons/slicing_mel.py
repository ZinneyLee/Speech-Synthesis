import torch
import numpy as np

def rand_slice_segments(mel, mel_lengths=None, segment_size=32):

    b, t = mel.size()
    if mel_lengths is None:
        mel_lengths = t
    ids_str_max = mel_lengths - segment_size

    ids_str = np.random.randint(0, ids_str_max)
    ret = mel[ids_str:ids_str+segment_size]

    return ret, ids_str




'''
[Reference code: VITS]

def slice_segments(x, ids_str, segment_size=4):
  ret = torch.zeros_like(x[:, :, :segment_size])
  for i in range(x.size(0)):
    idx_str = ids_str[i]
    idx_end = idx_str + segment_size
    ret[i] = x[i, :, idx_str:idx_end]
  return ret


def rand_slice_segments(x, x_lengths=None, segment_size=4):
  b, d, t = x.size()
  if x_lengths is None:
    x_lengths = t
  ids_str_max = x_lengths - segment_size + 1
  ids_str = (torch.rand([b]).to(device=x.device) * ids_str_max).to(dtype=torch.long)
  ret = slice_segments(x, ids_str, segment_size)
  return ret, ids_str
'''