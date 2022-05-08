import torch.nn as nn
from math import sqrt
import torch
import numpy as np
import torch.nn.functional as F
# compute atten
def get_score(p, q):
    sim_matrix = p.matmul(q.transpose(-2,-1))
    a = torch.norm(p, p=2,dim=-1)
    b = torch.norm(q, p=2,dim=-1)
    sim_matrix /= a.unsqueeze(-1)
    sim_matrix /= b.unsqueeze(-2)
    return sim_matrix

def compute_cos_atten(p, q):
    weight = get_score(p, q)
    weight = F.softmax(weight, dim=1)
    weight = weight.unsqueeze(2)
    atten = q * weight
    atten = atten.sum(dim=1)
    atten = atten.mean(dim=0)
    return atten

def compute_indot_atten(p, q):
    sim_matrix = p.matmul(q.transpose(-2,-1))
    weight = F.softmax(sim_matrix, dim = 1)
    weight = weight.unsqueeze(2)
    atten = q * weight
    atten = atten.sum(dim=1)
    atten = atten.mean(dim=0)
    return atten
def compute_atten(select,d_encode, w_encode, n_d_encode, n_w_encode):
    new_d_encode = []
    new_w_encode = []
    l = len(d_encode)
    for i in range(l):
        name_code = torch.Tensor(n_d_encode[i])
        time_code = torch.Tensor(d_encode[i])
        if select == 0:
            new_code = compute_cos_atten(time_code, name_code)
        elif select == 1:
            new_code = compute_indot_atten(time_code, name_code)
        elif select == 2:
            new_code = compute_cos_atten(name_code, time_code)
        elif select == 3:
            new_code = compute_cos_atten(time_code, name_code)
        new_d_encode.append(new_code.tolist())
    
    for i in range(l):
        name_code = torch.Tensor(n_w_encode[i])
        time_code = torch.Tensor(w_encode[i])
        if select == 0:
            new_code = compute_cos_atten(time_code, name_code)
        elif select == 1:
            new_code = compute_indot_atten(time_code, name_code)
        elif select == 2:
            new_code = compute_cos_atten(name_code, time_code)
        elif select == 3:
            new_code = compute_cos_atten(time_code, name_code)
        new_w_encode.append(new_code.tolist())

    return new_d_encode, new_w_encode