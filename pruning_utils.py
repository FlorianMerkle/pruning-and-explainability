import torch
from torch.nn.utils import prune

def get_prunable_modules(model):
    modules_to_prune = []
    for i, m in enumerate(list(model.named_modules())):
        if isinstance(m[1], torch.nn.Conv2d):
            modules_to_prune.append(model.get_submodule(m[0]))
    return modules_to_prune

def get_sparsity(modules):
    sparsities = []
    for m in modules:
        sparsities.append(((m.weight == 0).sum()/m.weight.numel()).item())
    sparsities = torch.Tensor(sparsities)
    return {
        'min':sparsities.min().item(), 
        'max':sparsities.max().item(), 
        'mean':sparsities.mean().item()}

def l1_prune(modules, amount):
    for m in modules:
        prune.L1Unstructured(.0).apply(m, 'weight', amount)
    return get_sparsity(modules)