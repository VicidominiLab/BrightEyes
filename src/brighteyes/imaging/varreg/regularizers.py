import torch

def grad_l2(obj_set):

    obj_mean = obj_set.mean(dim=-1, keepdim=True)

    u = 2 * obj_mean
    v = 2 * obj_set

    return u, v


def grad_l1(obj_set):

    obj_mean = obj_set.mean(dim=-1, keepdim=True)
    eps = torch.finfo(torch.float32).eps

    denominator  = torch.sqrt( (obj_set - obj_mean)**2 + eps**2 )

    u = obj_mean / denominator
    v = obj_set / denominator

    return u, v


def grad_kl(obj_set):

    obj_mean = obj_set.mean(dim=-1, keepdim=True)
    eps = torch.finfo(torch.float32).eps

    u = obj_mean / (obj_set + eps)
    v = 1

    return u, v