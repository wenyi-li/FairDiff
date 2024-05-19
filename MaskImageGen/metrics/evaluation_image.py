import os
import torch
import sys
import numpy as np
import warnings
from scipy.stats import entropy
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
from scipy.optimize import linear_sum_assignment
from torch.nn.functional import cosine_similarity
sys.path.append("../")

def torch_cos(a,b):
    d = torch.mul(a, b)
    cos = torch.sum(d, dim=1)
    return cos

def _pairwise_cosine_second(sample_img, ref_img, batch_size):
    N_sample = sample_img.shape[0]
    N_ref = ref_img.shape[0]
    assert N_sample == N_ref, "REF:%d SMP:%d" % (N_ref, N_sample)
    all_cosine = []
    iterator = range(N_sample)

    for sample_b_start in iterator:
        sample_batch = sample_img[sample_b_start].float().unsqueeze(0)
        cosine_lst = []
        for ref_b_start in range(0, N_ref, batch_size):
            ref_b_end = min(N_ref, ref_b_start + batch_size)
            ref_batch = ref_img[ref_b_start:ref_b_end].float()
            batch_size_ref = ref_batch.size(0)
            
            sample_batch = torch.div(sample_batch,torch.norm(sample_batch, dim=1).reshape(-1,1))
            ref_batch = torch.div(ref_batch,torch.norm(ref_batch, dim=1).reshape(-1,1))
            cosine = torch_cos(sample_batch,ref_batch)
            cosine = (1 - cosine) / 2
            cosine_lst.append(cosine.reshape(1, batch_size_ref))
        cosine_lst = torch.cat(cosine_lst, dim=1)
        all_cosine.append(cosine_lst)

    all_cosine = torch.cat(all_cosine, dim=0)  # N_sample, N_ref

    return all_cosine

def lgan_mmd_cov(all_dist):
    N_sample, N_ref = all_dist.size(0), all_dist.size(1)
    min_val_fromsmp, min_idx = torch.min(all_dist, dim=1)
    min_val, _ = torch.min(all_dist, dim=0)
    
    sorted_min_val, sorted_indices = torch.sort(min_val, descending=True)
    mmd = min_val.mean()
    mmd_smp = min_val_fromsmp.mean()
    cov = float(min_idx.unique().view(-1).size(0)) / float(N_ref)
    cov = torch.tensor(cov).to(all_dist)
    return {
        'lgan_mmd': mmd,
        'lgan_cov': cov
    }

def compute_all_metrics(sample_pcs, ref_pcs, batch_size, accelerated_cd=False):
    results = {}
    M_rs_cosine = _pairwise_cosine_second(sample_pcs, ref_pcs, batch_size)
    res_cosine = lgan_mmd_cov(M_rs_cosine.t())
    results.update({
        "%s-cosine" % k: v for k, v in res_cosine.items()
    })

    M_rr_cosine = _pairwise_cosine_second(ref_pcs, ref_pcs, batch_size)
    M_ss_cosine = _pairwise_cosine_second(sample_pcs, sample_pcs, batch_size)

    return results
