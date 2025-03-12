import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from numba import jit
from torch.autograd import Function


def ContrastiveSigmoid(v_emb,t_emb,t_prime,b,only_negatives=False):
    # v_emb : video model embedding [n, dim]
    # t_emb : text model embedding [n, dim]
    # t_prime, b : learnable temperature and bias
    # n : mini-batch size
    n = v_emb.size(0)
    n = torch.tensor(n).to(b.device)
    
    logSigmoid = nn.LogSigmoid()
    t = torch.exp(t_prime)
    z_v = F.normalize(v_emb,dim=-1)
    z_t = F.normalize(t_emb,dim=-1)
    logits = torch.matmul(z_v, z_t.T) * t + b
    
    if only_negatives:
        labels = torch.zeros(n,n) - torch.ones(n) # all -1 
        labels = labels.to(b.device)
    else:
        labels = 2 * torch.eye(n) - torch.ones(n) # -1 with diagonal 1
        labels = labels.to(b.device)

    loss = -torch.sum(logSigmoid(labels * logits)) / n
    return loss

        
    

if __name__ == "__main__":
    
    pass