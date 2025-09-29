import pesudo_tree
import cupy as cp
import torch
import nvtx

print(pesudo_tree.add(1, 2))
print(cp.add(1, 2))
print(torch.add(1, 2))
