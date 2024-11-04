import numpy as np
import time
import torch
import math

from sympy import false

from env import Env
from datasets import generate_circles_dataset, generate_RRTStar_dataset, generate_RRTStar_dataset_fixed, \
    PathPlanningDataset
from diffusion import PathDenoiser, evaluate, train_model
from diffusers.utils.torch_utils import randn_tensor

from image import Image
from image import color_palette


def evaluate_steering_combinations(dir_path, test_name, n, inference_steps, starts, goals, env, model):
    time_start = time.time()
    paths = model.generate_paths(n, inference_steps, starts, goals, steer_length=False, steer_obstacles=False)
    img = Image(f"{dir_path}/{test_name}_no_steering.svg", env)
    img.add_paths(paths)
    print(f"STEERING TEST ({test_name}) - no steering: {time.time() - time_start}")

    time_start = time.time()
    paths = model.generate_paths(n, inference_steps, starts, goals, steer_length=True, steer_obstacles=False)
    img = Image(f"{dir_path}/{test_name}_length_steering.svg", env)
    img.add_paths(paths)
    print(f"STEERING TEST ({test_name}) - length steering: {time.time() - time_start}")

    time_start = time.time()
    paths = model.generate_paths(n, inference_steps, starts, goals, steer_length=False, steer_obstacles=True)
    img = Image(f"{dir_path}/{test_name}_obstacle_steering.svg", env)
    img.add_paths(paths)
    print(f"STEERING TEST ({test_name}) - obstacle steering: {time.time() - time_start}")

    time_start = time.time()
    paths = model.generate_paths(n, inference_steps, starts, goals, steer_length=True, steer_obstacles=True)
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

    dataset = PathPlanningDataset("../datasets/datasetRRT_hard.pt")
    #
    # # dataset image
    # Image("../out/hard_test/hard_env.svg", env)
    # img = Image("../out/hard_test/hard_dataset.svg", env)
    # for i in range(0, len(dataset), math.ceil(len(dataset) / 30)):
    #     img.add_path(dataset[i].T)

    # train_model("../models/hard_model_1000.pt", dataset)

    model = PathDenoiser("../models/hard_model_1000.pt", env)
    n = 10
    start = [0.1, 0.1]
    goal = [0.9, 0.9]
    evaluate_steering_combinations("../out/hard_test", "hard", n, 300,
                                   [start for _ in range(n)], [goal for _ in range(n)], env, model)


def test_easy():
    """
    Simple test with one obstacle in the middle.
    """
    env = Env(1.0, 1.0)
    env.add_obstacle([(0.32, 0.37), (0.29, 0.56), (0.42, 0.68), (0.58, 0.69), (0.70, 0.52), (0.66, 0.35), (0.49, 0.29)])

    # generate_RRTStar_dataset("../datasets/datasetRRT_easy_10000.pt", 200, 50, 32, env, 0.2)

    # dataset = PathPlanningDataset("../datasets/datasetRRT_easy_10000.pt")
    # # dataset image
    # Image("../out/easy_env.svg", env)
    # img = Image("../out/easy_dataset.svg", env)
    # for i in range(0, len(dataset), math.ceil(len(dataset) / 30)):
    #     img.add_path(dataset[i].T)

    # train_model("../models/easy_model_10000.pt", dataset)

    model = PathDenoiser("../models/easy_model_10000.pt", env)
    n = 3
    start = [0.1, 0.5]
    goal = [0.9, 0.5]
    # evaluate_steering_combinations("easy", n, 300, [start for _ in range(n)], [goal for _ in range(n)], env, model)
    seed_generator = torch.manual_seed(0)
    noise_shape = (n, 2, 32)
    noise_input = randn_tensor(noise_shape, generator=seed_generator, device=model.model.device)
    no_steering = np.transpose(
        evaluate(model.model, model.noise_scheduler, noise_input, seed_generator, model.model.device,
                 env, 300, None, True, True, "../out/easy_diffusion").cpu().detach().numpy(), (0, 2, 1))


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

    # model_r_a_s = PathDenoiser("../models/circles_model_rotated_and_swapped_2000.pt", env)
    # no_steering_r_a_s = model_r_a_s.generate_random_paths(10, 300)
    # img = Image("../out/circles_test/circles_inference_rotated_and_swapped_no_steering.svg", env)
    # img.add_paths(no_steering_r_a_s)
    # length_steering_r_a_s = model_r_a_s.generate_random_paths(10, 300, True)
    # img = Image("../out/circles_test/circles_inference_rotated_and_swapped_length_steering.svg", env)
    # img.add_paths(length_steering_r_a_s)



def test_empty():
    env = Env(1.0, 1.0)
    # generate_RRTStar_dataset("../datasets/datasetRRT_empty_1000.pt", 200, 5, 32, env, 0.1)

    dataset = PathPlanningDataset("../datasets/datasetRRT_empty_1000.pt")
    # # dataset image
    # Image("../out/empty_env.svg", env)
    # img = Image("../out/empty_dataset.svg", env)
    # for i in range(0, len(dataset), math.ceil(len(dataset) / 30)):
    #     img.add_path(dataset[i].T)

    # train_model("../models/easy_empty_1000.pt", dataset)

    model = PathDenoiser("../models/easy_empty_1000.pt", env)
    n = 10
    start = [0.1, 0.5]
    goal = [0.9, 0.5]
    evaluate_steering_combinations("empty", n, [start for _ in range(n)], [goal for _ in range(n)], env, model)


if __name__ == "__main__":
    # test_hard()
    # test_easy()
    test_circles()
    # test_empty()
