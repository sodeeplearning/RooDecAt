from transformer.torch_imp import RooDecAtLM
import torch

model = RooDecAtLM()

print(model.generate([1, 2, 3], use_cuda=False))
