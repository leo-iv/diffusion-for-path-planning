import torch
import numpy as np

from .utils import generate_start_goal

def _resample_path(path, new_n_points):
    # resamples given path so it has new_n_points points
    total_path_len = 0.0

    for i in range(len(path) - 1):
        total_path_len += np.linalg.norm(np.array(path[i + 1]) - np.array(path[i]))

    delta = total_path_len / (new_n_points - 1)
    resampled_path = np.zeros((new_n_points, 2))

    point_counter = 0
    segment_idx = 0
    while point_counter < new_n_points and segment_idx < len(path - 1):
        segment_start = path[segment_idx]
        segment_stop = path[segment_idx + 1]

        direction = segment_stop - segment_start
        distance = np.linalg.norm(direction)
        direction = (1.0 / distance) * direction  # normalizing to 1.0 norm

        iters = int(distance / delta)
        for i in range(iters + 1):
            resampled_path[point_counter] = segment_start + ((i * delta) * direction)
            point_counter += 1

            if point_counter >= new_n_points:
                break

        segment_idx += 1
    return resampled_path


def generate_RRTStar_dataset(file_name, n_starts, samples_per_start, path_length, env, rrt_star_time=0.2):
    """
    Generates a dataset of 2D RRT* generated paths and saves it into .pt file - can be then loaded by the PathPlanningDataset class.

    Args:
        file_name: .pt file
        n_starts: number of randomly generated start and goal configuration
        samples_per_start: number of generated paths from one start configuration
        path_length: number of nodes on the path
        env: instance of the env.Env class
        rrt_star_time: max time for the RRT* algorithm
    """
    n_samples = n_starts * samples_per_start
    dataset = torch.empty(n_samples, 2, path_length)

    for start_idx in range(n_starts):
        start, goal = generate_start_goal(env)

        for sample_idx in range(samples_per_start):
            path = env.run_RRTstar(start, goal, rrt_star_time)
            resampled_path = _resample_path(path, path_length)

            dataset[start_idx * samples_per_start + sample_idx, :, :] = torch.tensor(resampled_path.T)

    torch.save(dataset, file_name)


def generate_RRTStar_dataset_fixed(file_name, starts, goals, samples_per_start, path_length, env, rrt_star_time=0.2):
    """
    Generates a dataset of 2D RRT* generated paths and saves it into .pt file - can be then loaded by the PathPlanningDataset class.
    Same as generate_RRTStar_dataset, but the start and goal configurations are given and not randomly sampled.
    Args:
        file_name: .pt file
        starts: sequence of (x, y) start configurations
        samples_per_start: number of generated paths from one start configuration to the corresponding goal
        path_length: number of nodes on the path
        env: instance of the env.Env class
        rrt_star_time: max time for the RRT* algorithm
    """
    assert len(starts) <= len(goals), "Not enough goal configurations given to generate_RRTStar_dataset_fixed"

    n_samples = len(starts) * samples_per_start
    dataset = torch.empty(n_samples, 2, path_length)

    for i in range(len(starts)):
        start = starts[i]
        goal = goals[i]

        for sample_idx in range(samples_per_start):
            path = env.run_RRTstar(start, goal, rrt_star_time)
            resampled_path = _resample_path(path, path_length)

            dataset[i * samples_per_start + sample_idx, :, :] = torch.tensor(resampled_path.T)

    torch.save(dataset, file_name)
