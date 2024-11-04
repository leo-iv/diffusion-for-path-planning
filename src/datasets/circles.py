import math
import random

import numpy as np
import torch
from numpy.ma.core import angle


def _generate_circular_path(center, radius, path_length, start_angle, clockwise=True):
    path = torch.empty((2, path_length))

    angle_delta = (2 * math.pi / (path_length - 1)) * (1 if clockwise else -1)
    for i in range(path_length):
        angle = i * angle_delta + start_angle
        path[:, i] = radius * torch.tensor([math.cos(angle), math.sin(angle)]) + torch.tensor(center)

    return path


def generate_circles_dataset(file_name, n_circles, center, radius_min, radius_max, path_length, random_start=False,
                             change_directions=False):
    """
    Generates a dataset of 2D circular paths. Result dataset is saved into .pt file - can be then loaded by the PathPlanningDataset class.

    Args:
        file_name: .pt file
        n_circles: number of circles in the dataset
        center: common center point for all circles - (x, y) tuple
        radius_min: radius for the smallest possible circle in the dataset
        radius_max: radius for the biggest possible circle in the dataset
        path_length: number of nodes on the path
        random_start: if set to True, the start point of the circular path is randomly chosen
        change_directions: if set to True, the direction (clockwise / counter-clockwise) is randomly chosen for each circle
    """
    dataset = torch.empty((n_circles, 2, path_length))
    for i in range(n_circles):
        start_angle = random.uniform(0.0, 2 * np.pi) if random_start else 0.0
        clockwise_direction = random.choice([True, False]) if change_directions else True
        dataset[i, :, :] = _generate_circular_path(center, random.uniform(radius_min, radius_max), path_length,
                                                   start_angle, clockwise_direction)

    torch.save(dataset, file_name)
