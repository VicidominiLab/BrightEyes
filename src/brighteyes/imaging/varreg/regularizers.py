import torch

from .loss import compute_centroid

def grad_l2(obj_set):

    obj_mean = compute_centroid(obj_set, 'l2')

    u = 2 * obj_mean
    v = 2 * obj_set

    return u, v


def grad_l1(obj_set):

    obj_mean = compute_centroid(obj_set, 'l1')

    denominator  = torch.sqrt( (obj_set - obj_mean)**2 + eps**2 )

    u = obj_mean / denominator
    v = obj_set / denominator

    return u, v


def grad_kl(obj_set):

    obj_mean = compute_centroid(obj_set, 'kl')

    u = obj_mean / (obj_set + eps)
    v = 1

    return u, v