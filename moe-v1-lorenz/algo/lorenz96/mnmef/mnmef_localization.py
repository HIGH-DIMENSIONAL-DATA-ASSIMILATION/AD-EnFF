"""Periodic-distance localization utilities for Lorenz-96."""

import torch


def pairwise_distances(first, second=None, domain=None):
    second = first if second is None else second
    first = torch.atleast_2d(torch.as_tensor(first))
    second = torch.atleast_2d(torch.as_tensor(second))
    difference = first[:, None] - second
    if domain is not None:
        domain = torch.as_tensor(domain).reshape(1, 1, -1)
        difference = difference.abs()
        difference = torch.minimum(difference, domain - difference)
    return difference.square().sum(-1).sqrt()


def create_loc_mat(distance_weights, distance_values, pairwise_distance):
    expanded_distances = pairwise_distance[None, None]
    mask = expanded_distances == distance_values[None, :, None, None]
    return (mask * distance_weights[:, :, None, None]).sum(1)
