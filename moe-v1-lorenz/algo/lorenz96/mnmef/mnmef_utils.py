"""Ensemble-centering utilities adapted from DALearning."""
import torch


def center(E, axis=1, rescale=False):
    x = torch.mean(E, dim=axis, keepdims=True)
    X = E - x

    if rescale:
        N = E.shape[axis]
        X *= torch.sqrt(torch.tensor(N / (N - 1))).to(E.device)

    return X, x

def mean0(E, axis=1, rescale=True):
    return center(E, axis=axis, rescale=rescale)[0]
