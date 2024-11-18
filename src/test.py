import random

import numpy as np
import time
import torch
import math

from env import Env
from datasets import generate_circles_dataset, generate_RRTStar_dataset, generate_RRTStar_dataset_fixed, \
    PathPlanningDataset
from diffusion import PathDenoiser, evaluate, train_model
from diffusers.utils.torch_utils import randn_tensor

from image import Image
from image import color_palette


def create_dataset_img(filename, dataset, n_paths, env):
    img = Image(filename, env)
    img.add_paths(np.array([random.choice(dataset).T for _ in range(n_paths)]))


def evaluate_steering_combinations(dir_path, test_name, starts, goals, env, model, samples_per_start=1,
                                   inference_steps=300):
    time_start = time.time()
    paths = model.generate_paths(starts, goals, samples_per_start, inference_steps, steer_length=False,
                                 steer_obstacles=False)
    img = Image(f"{dir_path}/{test_name}_no_steering.svg", env)
    img.add_paths(paths)
    print(f"STEERING TEST ({test_name}) - no steering: {time.time() - time_start}")

    time_start = time.time()
    paths = model.generate_paths(starts, goals, samples_per_start, inference_steps, steer_length=True,
                                 steer_obstacles=False)
    img = Image(f"{dir_path}/{test_name}_length_steering.svg", env)
    img.add_paths(paths)
    print(f"STEERING TEST ({test_name}) - length steering: {time.time() - time_start}")

    time_start = time.time()
    paths = model.generate_paths(starts, goals, samples_per_start, inference_steps, steer_length=False,
                                 steer_obstacles=True)
    img = Image(f"{dir_path}/{test_name}_obstacle_steering.svg", env)
    img.add_paths(paths)
    print(f"STEERING TEST ({test_name}) - obstacle steering: {time.time() - time_start}")

    time_start = time.time()
    paths = model.generate_paths(starts, goals, samples_per_start, inference_steps, steer_length=True,
                                 steer_obstacles=True)
    img = Image(f"{dir_path}/{test_name}_obstacle_and_length_steering.svg", env)
    img.add_paths(paths)
    print(f"STEERING TEST ({test_name}) - obstacle and length steering: {time.time() - time_start}")


def test_hard():
    env = Env(1.0, 1.0)
    env.add_obstacle([(0.35, 0.11), (0.45, 0.22), (0.53, 0.06)])
    env.add_obstacle([(0.12, 0.30), (0.16, 0.48), (0.26, 0.37)])
    env.add_obstacle([(0.69, 0.22), (0.90, 0.40), (0.82, 0.16)])
    env.add_obstacle([(0.43, 0.42), (0.57, 0.33), (0.73, 0.42), (0.70, 0.58), (0.50, 0.60)])
    env.add_obstacle([(0.14, 0.65), (0.30, 0.60), (0.47, 0.75), (0.27, 0.71), (0.14, 0.80)])
    env.add_obstacle([(0.66, 0.71), (0.88, 0.73), (0.50, 0.90)])

    # generate_RRTStar_dataset("../datasets/datasetRRT_hard_1000.pt", 200, 5, 32, env, 0.2)

    # dataset = PathPlanningDataset("../datasets/datasetRRT_hard.pt")
    #
    # # dataset image
    # Image("../out/hard_test/hard_env.svg", env)
    # img = Image("../out/hard_test/hard_dataset.svg", env)
    # for i in range(0, len(dataset), math.ceil(len(dataset) / 30)):
    #     img.add_path(dataset[i].T)

    # train_model("../models/hard_model_1000.pt", dataset)

    # model = PathDenoiser("../models/hard_model_1000.pt", env)
    # n = 10
    # start = [0.1, 0.1]
    # goal = [0.9, 0.9]
    # evaluate_steering_combinations("../out/hard_test", "hard", n, 300,
    #                                [start for _ in range(n)], [goal for _ in range(n)], env, model)

    # file_prefix = "../out/hard_test/fixed_ends/"
    # # generate_RRTStar_dataset_fixed("../datasets/datasetRRT_hard_fixed_1000.pt", [(0.1, 0.1)], [(0.9, 0.9)], 1000, 32, env)
    # dataset = PathPlanningDataset("../datasets/datasetRRT_hard_fixed_1000.pt")
    # # create_dataset_img(file_prefix + "dataset.svg", dataset, 30, env)
    # # train_model("../models/hard_model_fixed_ends_1000.pt", dataset)
    # model = PathDenoiser("../models/hard_model_fixed_ends_1000.pt", env)
    # evaluate_steering_combinations(file_prefix, "hard_env_fixed_ends", [(0.1, 0.1)], [(0.9, 0.9)], env, model, 10)

    # one direction straight
    dataset_size = 1000
    starts = [(random.uniform(0.05, 0.95), 0.05) for _ in range(dataset_size)]
    goals = [(start_x, 0.95) for (start_x, start_y) in starts]
    # generate_RRTStar_dataset_fixed("../datasets/datasetRRT_hard_one_direction_straight_1000.pt",
    #                                starts,
    #                                goals,
    #                                1, 32, env)
    file_prefix = "../out/hard_test/one_direction_straight/"
    dataset = PathPlanningDataset("../datasets/datasetRRT_hard_one_direction_straight_1000.pt")
    # dataset image
    # create_dataset_img(file_prefix + "dataset.svg", dataset, 30, env)

    train_model("../models/hard_one_direction_straight_1000.pt", dataset)
    model = PathDenoiser("../models/hard_one_direction_straight_1000.pt", env)
    n = 10
    start = [0.5, 0.05]
    goal = [0.5, 0.95]
    evaluate_steering_combinations(file_prefix, "hard_one_direction_straight", [start for _ in range(n)],
                                   [goal for _ in range(n)],
                                   env, model)


def test_easy():
    """
    Simple test with one obstacle in the middle.
    """
    env = Env(1.0, 1.0)
    env.add_obstacle([(0.32, 0.37), (0.29, 0.56), (0.42, 0.68), (0.58, 0.69), (0.70, 0.52), (0.66, 0.35), (0.49, 0.29)])
    # Image("../out/easy_test/easy_env.svg", env)

    # generic dataset test:
    # generate_RRTStar_dataset("../datasets/datasetRRT_easy_10000.pt", 200, 50, 32, env, 0.2)
    # dataset = PathPlanningDataset("../datasets/datasetRRT_easy_10000.pt")
    # create_dataset_img("../out/easy_test/generic_dataset/dataset.svg", dataset, 30, env)

    # img = Image("../out/easy_test/generic_dataset/dataset.svg", env)
    # for i in range(0, len(dataset), math.ceil(len(dataset) / 30)):
    #     img.add_path(dataset[i].T)

    # train_model("../models/easy_model_10000.pt", dataset)

    # model = PathDenoiser("../models/easy_model_10000.pt", env)
    # start = [0.1, 0.5]
    # goal = [0.9, 0.5]
    # evaluate_steering_combinations("../out/easy_test/generic_dataset", "easy_generic", 10, 300,
    #                                [start for _ in range(10)], [goal for _ in range(10)], env, model)
    # model.generate_paths(3, 300, [start for _ in range(3)], [goal for _ in range(3)], True, True,
    #                      "../out/easy_test/generic_dataset/diffusion_process")

    # fixed start and goal dataset:
    file_prefix = "../out/easy_test/fixed_ends_dataset/"
    # start = [0.1, 0.5]
    # goal = [0.9, 0.5]
    # generate_RRTStar_dataset_fixed("../datasets/datasetRRT_easy_fixed_start_and_goal_2000.pt",
    #                                [start], [goal], 2000, 32, env)
    # dataset = PathPlanningDataset("../datasets/datasetRRT_easy_fixed_start_and_goal_2000.pt")
    # create_dataset_img("../out/easy_test/fixed_ends_dataset/dataset.svg", dataset, 30, env)
    # train_model("../models/easy_model_fixed_start_and_goal_2000.pt", dataset)
    # model = PathDenoiser("../models/easy_model_fixed_start_and_goal_2000.pt", env)
    # evaluate_steering_combinations("../out/easy_test/fixed_ends_dataset", "easy_fixed_ends", 10, 300,
    #                                [start for _ in range(10)], [goal for _ in range(10)], env, model)
    # paths = model.generate_paths(9, 300, [(0.1, 0.1 * y) for y in range(1, 10)], [(0.9, 0.1 * y) for y in range(1, 10)],
    #                              True, True)
    # Image(file_prefix + "different_ends.svg", env).add_paths(paths)

    # one direction test:
    file_prefix = "../out/easy_test/one_direction/"
    dataset_size = 1000
    # generate_RRTStar_dataset_fixed(f"../datasets/datasetRRT_easy_one_direction_{dataset_size}.pt",
    #                                [(random.uniform(0.05, 0.25), random.uniform(0.1, 0.9)) for _ in
    #                                 range(dataset_size)],
    #                                [(random.uniform(0.75, 0.95), random.uniform(0.1, 0.9)) for _ in
    #                                 range(dataset_size)],
    #                                1, 32, env)
    # dataset = PathPlanningDataset(f"../datasets/datasetRRT_easy_one_direction_{dataset_size}.pt")
    # create_dataset_img(file_prefix + "dataset.svg", dataset, 40, env)
    # train_model("../models/easy_model_one_direction_1000.pt", dataset)
    # model = PathDenoiser("../models/easy_model_one_direction_1000.pt", env)
    # evaluate_steering_combinations(file_prefix, "easy_one_direction", [(0.1, 0.1 * y) for y in range(1, 10)],
    #                                [(0.9, 0.1 * y) for y in range(1, 10)], env, model)
    # paths = model.generate_paths(10, 300, [(0.1, 0.5) for _ in range(10)], [(0.9, 0.5) for _ in range(10)],
    #                              True, True)
    # Image(file_prefix + "easy_one_direction_fixed_start.svg", env).add_paths(paths)

    # one direction straight test:
    file_prefix = "../out/easy_test/one_direction_straight/"
    dataset_size = 2000
    starts = [(0.1, random.uniform(0.05, 0.95)) for _ in range(int(dataset_size / 2))]
    goals = [(0.9, start_y) for (_, start_y) in starts]
    # generate_RRTStar_dataset_fixed(f"../datasets/datasetRRT_easy_one_direction_straight_{dataset_size}.pt",
    #                                starts, goals, 2, 32, env)
    dataset = PathPlanningDataset(f"../datasets/datasetRRT_easy_one_direction_straight_{dataset_size}.pt")
    # create_dataset_img(file_prefix + "dataset.svg", dataset, 50, env)
    # train_model("../models/easy_model_one_direction_straight_2000.pt", dataset)
    model = PathDenoiser("../models/easy_model_one_direction_1000.pt", env)
    evaluate_steering_combinations(file_prefix, "easy_one_direction_straight",
                                   [(0.1, 0.1 * y) for y in range(1, 10)], [(0.9, 0.1 * y) for y in range(1, 10)], env,
                                   model, samples_per_start=2)
    paths = model.generate_paths([(0.1, 0.1)], [(0.9, 0.1)], 10, 300, True, True, file_prefix + "diffusion_process")
    Image(file_prefix + "easy_one_direction_straight_fixed_start.svg", env).add_paths(paths)

    paths = model.generate_paths([(0.9, 0.1)], [(0.1, 0.1)], 10, 300, True, True)
    Image(file_prefix + "easy_one_direction_straight_opposite_direction.svg", env).add_paths(paths)


def test_circles():
    env = Env(1.0, 1.0)
    # generate_circles_dataset("../datasets/dataset_circles_aligned_2000.pt", 2000, (0.5, 0.5), 0.1, 0.5, 32)
    # generate_circles_dataset("../datasets/dataset_circles_rotated_2000.pt", 2000, (0.5, 0.5), 0.1, 0.5, 32, True)
    # generate_circles_dataset("../datasets/dataset_circles_swapped_2000.pt", 2000, (0.5, 0.5), 0.1, 0.5, 32, False, True)
    # generate_circles_dataset("../datasets/dataset_circles_rotated_and_swapped_2000.pt", 2000, (0.5, 0.5), 0.1, 0.5, 32,
    #                          True, True)

    dataset_aligned = PathPlanningDataset("../datasets/dataset_circles_aligned_2000.pt")
    dataset_rotated = PathPlanningDataset("../datasets/dataset_circles_rotated_2000.pt")
    dataset_swapped = PathPlanningDataset("../datasets/dataset_circles_swapped_2000.pt")
    dataset_rotated_and_swapped = PathPlanningDataset("../datasets/dataset_circles_rotated_and_swapped_2000.pt")

    # dataset images
    # img = Image("../out/circles_test/circles_dataset_aligned.svg", env)
    # for i in range(0, 10):
    #     img.add_path(dataset_aligned[i].T, color_palette[i])
    #
    # img = Image("../out/circles_test/circles_dataset_rotated.svg", env)
    # for i in range(0, 10):
    #     img.add_path(dataset_rotated[i].T, color_palette[i])

    # img = Image("../out/circles_test/circles_dataset_swapped.svg", env)
    # for i in range(0, 10):
    #     print(dataset_swapped[i].T)
    #     img.add_path(dataset_swapped[i].T, color_palette[i])

    # img = Image("../out/circles_test/circles_dataset_rotated_and_swapped.svg", env)
    # for i in range(0, 10):
    #     print(dataset_rotated_and_swapped[i].T)
    #     img.add_path(dataset_rotated_and_swapped[i].T, color_palette[i])

    # train_model("../models/circles_model_aligned_2000.pt", dataset_aligned)
    # train_model("../models/circles_model_rotated_2000.pt", dataset_rotated)
    # train_model("../models/circles_model_swapped_2000.pt", dataset_swapped)
    # train_model("../models/circles_model_rotated_and_swapped_2000.pt", dataset_rotated_and_swapped)

    model_aligned = PathDenoiser("../models/circles_model_aligned_2000.pt", env)
    # no_steering_aligned = model_aligned.generate_random_paths(10, 300)
    # img = Image("../out/circles_test/circles_inference_aligned_no_steering.svg", env)
    # img.add_paths(no_steering_aligned)
    # length_steering_aligned = model_aligned.generate_random_paths(10, 300, True)
    # img = Image("../out/circles_test/circles_inference_aligned_length_steering.svg", env)
    # img.add_paths(length_steering_aligned)

    # diffusion visualization
    model_aligned.generate_random_paths(3, 300, True, False, "../out/circles_test/diffusion_process")

    # model_rotated = PathDenoiser("../models/circles_model_rotated_2000.pt", env)
    # no_steering_rotated = model_rotated.generate_random_paths(10, 300)
    # img = Image("../out/circles_test/circles_inference_rotated_no_steering.svg", env)
    # img.add_paths(no_steering_rotated)
    # length_steering_rotated = model_rotated.generate_random_paths(10, 300, True)
    # img = Image("../out/circles_test/circles_inference_rotated_length_steering.svg", env)
    # img.add_paths(length_steering_rotated)

    # model_swapped = PathDenoiser("../models/circles_model_swapped_2000.pt", env)
    # no_steering_swapped = model_swapped.generate_random_paths(10, 300)
    # img = Image("../out/circles_test/circles_inference_swapped_no_steering.svg", env)
    # img.add_paths(no_steering_swapped)
    # length_steering_swapped = model_swapped.generate_random_paths(10, 300, True)
    # img = Image("../out/circles_test/circles_inference_swapped_length_steering.svg", env)
    # img.add_paths(length_steering_swapped)
    #
    # model_r_a_s = PathDenoiser("../models/circles_model_rotated_and_swapped_2000.pt", env)
    # no_steering_r_a_s = model_r_a_s.generate_random_paths(10, 300)
    # img = Image("../out/circles_test/circles_inference_rotated_and_swapped_no_steering.svg", env)
    # img.add_paths(no_steering_r_a_s)
    # length_steering_r_a_s = model_r_a_s.generate_random_paths(10, 300, True)
    # img = Image("../out/circles_test/circles_inference_rotated_and_swapped_length_steering.svg", env)
    # img.add_paths(length_steering_r_a_s)


def test_empty():
    env = Env(1.0, 1.0)
    # # generate_RRTStar_dataset("../datasets/datasetRRT_empty_1000.pt", 200, 5, 32, env, 0.1)
    # file_prefix = "../out/empty_test/generic_dataset/"
    # dataset = PathPlanningDataset("../datasets/datasetRRT_empty_1000.pt")
    # # dataset image
    # create_dataset_img(file_prefix + "dataset.svg", dataset, 10, env)
    #
    # # train_model("../models/easy_empty_1000.pt", dataset)
    # model = PathDenoiser("../models/easy_empty_1000.pt", env)
    # n = 10
    # start = [0.1, 0.5]
    # goal = [0.9, 0.5]
    # evaluate_steering_combinations(file_prefix, "empty_generic", [start for _ in range(n)], [goal for _ in range(n)], env, model)

    # one direction
    # dataset_size = 1000
    # generate_RRTStar_dataset_fixed("../datasets/datasetRRT_empty_one_direction_1000.pt",
    #                                [(random.uniform(0.05, 0.25), random.uniform(0.1, 0.9)) for _ in
    #                                 range(dataset_size)],
    #                                [(random.uniform(0.75, 0.95), random.uniform(0.1, 0.9)) for _ in
    #                                 range(dataset_size)],
    #                                1, 32, env)
    # file_prefix = "../out/empty_test/one_direction/"
    # dataset = PathPlanningDataset("../datasets/datasetRRT_empty_one_direction_1000.pt")
    # # dataset image
    # create_dataset_img(file_prefix + "dataset.svg", dataset, 10, env)
    #
    # train_model("../models/empty_one_direction_1000.pt", dataset)
    # model = PathDenoiser("../models/empty_one_direction_1000.pt", env)
    # n = 10
    # start = [0.1, 0.5]
    # goal = [0.9, 0.5]
    # evaluate_steering_combinations(file_prefix, "empty_one_direction", [start for _ in range(n)],
    #                                [goal for _ in range(n)],
    #                                env, model)

    # one direction straight
    dataset_size = 1000
    # starts = [(0.1, random.uniform(0.05, 0.95)) for _ in range(int(dataset_size / 2))]
    # goals = [(0.9, start_y) for (_, start_y) in starts]
    # generate_RRTStar_dataset_fixed("../datasets/datasetRRT_empty_one_direction_straight_1000.pt",
    #                                starts,
    #                                goals,
    #                                1, 32, env)
    file_prefix = "../out/empty_test/one_direction_straight/"
    dataset = PathPlanningDataset("../datasets/datasetRRT_empty_one_direction_straight_1000.pt")
    # dataset image
    create_dataset_img(file_prefix + "dataset.svg", dataset, 30, env)

    train_model("../models/empty_one_direction_straight_1000.pt", dataset)
    model = PathDenoiser("../models/empty_one_direction_straight_1000.pt", env)
    n = 10
    start = [0.1, 0.5]
    goal = [0.9, 0.5]
    evaluate_steering_combinations(file_prefix, "empty_one_direction_straight", [start for _ in range(n)],
                                   [goal for _ in range(n)],
                                   env, model)


if __name__ == "__main__":
    test_hard()
    # test_easy()
    # test_circles()
    # test_empty()
