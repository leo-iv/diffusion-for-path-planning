import numpy as np
from cryptography.hazmat.backends.openssl import backend

from env import Env
from dataset_rrt import DatasetRRT, generate_datasetRRT
from torch.utils.data import DataLoader
import torch

from image import Image


def test_hard():
    env = Env(1.0, 1.0)
    env.add_obstacle([(0.35, 0.11), (0.45, 0.22), (0.53, 0.06)])
    env.add_obstacle([(0.12, 0.30), (0.16, 0.48), (0.26, 0.37)])
    env.add_obstacle([(0.69, 0.22), (0.90, 0.40), (0.82, 0.16)])
    env.add_obstacle([(0.43, 0.42), (0.57, 0.33), (0.73, 0.42), (0.70, 0.58), (0.50, 0.60)])
    env.add_obstacle([(0.14, 0.65), (0.30, 0.60), (0.47, 0.75), (0.27, 0.71), (0.14, 0.80)])
    env.add_obstacle([(0.66, 0.71), (0.88, 0.73), (0.50, 0.90)])

    # generate_datasetRRT("../datasets/datasetRRT_hard.pt", 200, 5, 32, env, 0.2)

    dataset = DatasetRRT("../datasets/datasetRRT_hard.pt")

    Image("../out/hard_env.svg", env)
    img = Image("../out/hard_dataset.svg", env)
    for path in dataset:
        img.add_path(path.T)


def test_easy():
    """
    Simple test with one obstacle in the middle.
    """
    env = Env(1.0, 1.0)
    env.add_obstacle([(0.32, 0.37), (0.29, 0.56), (0.42, 0.68), (0.58, 0.69), (0.70, 0.52), (0.66, 0.35), (0.49, 0.29)])

    # generate_datasetRRT("../datasets/datasetRRT_easy.pt", 200, 5, 32, env, 0.2)

    dataset = DatasetRRT("../datasets/datasetRRT_easy.pt")

    # creating dataset image
    Image("../out/easy_env.svg", env)
    img = Image("../out/easy_dataset.svg", env)
    for path in dataset:
        img.add_path(path.T)


if __name__ == "__main__":
    # test_hard()
    test_easy()
