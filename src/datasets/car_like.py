import torch
from tqdm import tqdm
import numpy as np
from src.car_like import move_car_like

from .utils import generate_start_goal


def _double_path_size(path, action_time, car_length):
    new_path = np.empty((len(path) * 2 - 1, 4))

    for i in range(len(path) - 1):
        new_path[2 * i, :] = path[i, :]
        new_path[2 * i + 1, :] = np.append(move_car_like(path[i, :3], path[i, 3], car_length, action_time / 2),
                                           path[i, 3])
    new_path[-1, :] = path[-1, :]

    return new_path


def _resample_path(path, new_n_points, action_time, car_length):
    while len(path) < new_n_points:
        path = _double_path_size(path, action_time, car_length)
        action_time = action_time / 2

    return path[:new_n_points, :]  # remove remaining elements

def _get_start_rotation(start, goal):
    raise NotImplementedError


def generate_car_like_dataset(file_name, env, n_samples, path_length=32, max_iters=1000,
                              steer_limit=np.pi / 4, n_actions=5, action_time=0.05, car_length=0.1,
                              n_collision_checks=3, goal_bias=0.1, goal_tolerance=0.05):
    """
    Generates a dataset of 2D car-like paths and saves it into .pt file - can be then loaded by the PathPlanningDataset class.

    Args:
        file_name: .pt file
        env: instance of the env.Env class
        n_samples: number of randomly generated start and goal configuration
        path_length: number of nodes on the final path (can result in cutting off some of the final nodes on the path)
        max_iters: max iters of the car-like RRT algorithm
        steer_limit: defines maximal car steering angle - steering angle is then generated from the interval
                         [-steer_limit, steer_limit] - python float
        n_actions: number of actions (steering angles) tried during tree expansion
        action_time: time per one action ( = time between nodes on the path)
        car_length: length of the car-like model
        n_collision_checks: number of collision checks performed on one tree expansion
        goal_bias: probability of expanding towards goal instead of sampling randomly
        goal_tolerance: defines the "goal region" around the goal position - float
    """
    dataset = torch.empty(n_samples, 4, path_length)

    for i in tqdm(range(n_samples), desc="Generating car-like dataset"):
        path = None
        while path is None:
            start, goal = generate_start_goal(env)
            start = np.append(start, np.random.uniform(0, 2 * np.pi))  # random start rotation
            path, _ = env.run_RRT_car_like(start, goal, max_iters=max_iters, steer_limit=steer_limit,
                                        n_actions=n_actions, action_time=action_time, car_length=car_length,
                                        n_collision_checks=n_collision_checks, goal_bias=goal_bias,
                                        goal_tolerance=goal_tolerance)

        resampled_path = _resample_path(path, path_length, action_time, car_length)
        dataset[i, :, :] = torch.tensor(resampled_path.T)

    torch.save(dataset, file_name)
