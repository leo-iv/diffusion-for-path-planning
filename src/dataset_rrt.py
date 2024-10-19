import torch
from torch.utils.data import Dataset
import random
import numpy as np


def __generate_start_goal(env):
    # generates free start and goal configurations in the environment at least half the env size apart
    width, height = env.size()

    def sample_start():
        # sample a random point from the "borders" of the env
        x = random.uniform(0, width / 2)
        y = random.uniform(0, height / 2)
        if x > width / 4:
            x += width / 2
        if y > height / 4:
            y += height / 2
        return x, y

    def sample_goal(start_coords):
        # sample from the opposite side of the env to start point
        start_x, start_y = start_coords
        if random.random() < 0.5:
            # "opposite x"
            if start_x < width / 4:
                x = random.uniform(3 * width / 4, width)
            else:
                x = random.uniform(0, width / 4)

            y = random.uniform(0, height)
        else:
            # "opposite y"
            if start_y < height / 4:
                y = random.uniform(3 * height / 4, height)
            else:
                y = random.uniform(0, height / 4)

            x = random.uniform(0, width)

        return x, y

    start = sample_start()
    while env.point_collides(start):
        start = sample_start()  # sample until we get collision free point

    stop = sample_goal(start)
    while env.point_collides(stop):
        stop = sample_goal(start)

    return start, stop


def __resample_path(path, new_n_points):
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


def generate_datasetRRT(file_name, n_starts, samples_per_start, path_length, env, rrt_star_time=0.2):
    """
    Generates a dataset of 2D RRT* generated paths and saves it into .pt file - can be then loaded by the DatasetRRT class.
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
        start, goal = __generate_start_goal(env)

        for sample_idx in range(samples_per_start):
            path = env.run_RRTstar(start, goal, rrt_star_time)
            resampled_path = __resample_path(path, path_length)

            dataset[start_idx * samples_per_start + sample_idx, :, :] = torch.tensor(resampled_path.T)

    torch.save(dataset, file_name)


class DatasetRRT(Dataset):
    """
    Dataset of 2D RRT* generated paths.
    """

    def __init__(self, file_name):
        """
        Args:
            file_name: .pt file generated by the generate_datasetRRT function
        """
        self.dataset = torch.load(file_name, weights_only=True)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)
