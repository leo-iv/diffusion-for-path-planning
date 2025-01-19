from env import load_hard_env, load_easy_env
from datasets.car_like import generate_car_like_dataset
from datasets import PathPlanningDataset
from diffusion import train_path_denoiser, TrainingConfig
from tree import Tree
from env import Env
from image import Image
import numpy as np
import random
import time
from diffusion import PathDenoiser


def test_nearest_neighbour():
    env = Env(1.0, 1.0)
    img = Image("../out/car_like_test/nn_test.svg", env)
    tree = Tree()
    tree.add_node(np.array([0.1, 0.1, 0.0]))
    tree.add_node(np.array([0.5, 0.5, 0.0]))
    tree.add_node(np.array([0.9, 0.9, 0.0]))
    tree.add_node(np.array([0.1, 0.9, 0.0]))

    for node in tree.nodes:
        img.add_circle(node.coords[:2], 0.01, (255, 0, 0))

    query = np.array([0.9, 0.6, 0.0])
    nearest = tree.get_nearest(query)

    img.add_circle(query[:2], 0.01, (0, 255, 0))
    img.add_star(nearest.coords[:2], 0.03, (0, 0, 255))


def test_rrt():
    env = Env(1.0, 1.0)
    start = np.array([0.1, 0.1, 0])
    goal = np.array([0.9, 0.9])
    start_time = time.time()
    path, tree = env.run_RRT_car_like(start, goal, goal_bias=0.00001, max_iters=1000)
    print(f"Empty: {time.time() - start_time} s")
    img = Image("../out/car_like_test/rrt_test/empty.svg", env)
    img.add_tree_car_like(tree)
    img.add_path_car_like(path)

    env = load_hard_env()
    start = np.array([0.1, 0.1, 0])
    goal = np.array([0.9, 0.9])
    path, tree = env.run_RRT_car_like(start, goal, n_collision_checks=5, max_iters=10000, n_actions=5)
    start_time = time.time()
    img = Image("../out/car_like_test/rrt_test/hard.svg", env)
    print(f"Hard: {time.time() - start_time} s")
    img.add_tree_car_like(tree)
    img.add_path_car_like(path)

    env = load_easy_env()
    start = np.array([0.1, 0.5, 0])
    goal = np.array([0.9, 0.5])
    start_time = time.time()
    path, tree = env.run_RRT_car_like(start, goal, n_collision_checks=5, max_iters=10000, n_actions=5)
    print(f"Easy: {time.time() - start_time} s")
    img = Image("../out/car_like_test/rrt_test/easy.svg", env)
    img.add_tree_car_like(tree)
    img.add_path_car_like(path)


def create_dataset_img(filename, dataset, n_paths, env):
    img = Image(filename, env)
    img.add_paths_car_like(np.array([random.choice(dataset).T for _ in range(n_paths)]))


def test_easy():
    env = load_easy_env()
    generate_car_like_dataset("../datasets/dataset_car_like_easy_50000.pt", env, 50000, path_length=32)
    dataset = PathPlanningDataset("../out/car_like_test/easy_50000/dataset_car_like_easy_50000.pt")
    create_dataset_img("../out/car_like_test/easy_50000/dataset.svg", dataset, 20, env)

    train_path_denoiser("../out/car_like_test/easy_50000/train", dataset,
                        eval_starts=np.array([(0.1, 0.5, float('Nan'), float('Nan')) for _ in range(10)]),
                        eval_goals=np.array([(0.9, 0.5, float('Nan'), float('Nan')) for _ in range(10)]),
                        env=env)


def test_hard():
    env = load_hard_env()

    generate_car_like_dataset("../out/car_like_test/hard_10000/dataset_car_like_hard_10000.pt", env, 50000, path_length=64)
    dataset = PathPlanningDataset("../out/car_like_test/hard_10000/dataset_car_like_hard_10000.pt")
    create_dataset_img("../out/car_like_test/hard_10000/dataset.svg", dataset, 20, env)

    train_path_denoiser("../out/car_like_test/hard_10000/train2", dataset,
                        config=TrainingConfig(sequence_length=64, in_channels=4, out_channels=4, num_epochs=100,
                                              learning_rate=1e-5, lr_warmup_steps=1000),
                        eval_starts=np.array([(0.9, 0.1, float('Nan'), float('Nan')) for _ in range(10)]),
                        eval_goals=np.array([(0.1, 0.9, float('Nan'), float('Nan')) for _ in range(10)]),
                        env=env)


if __name__ == "__main__":
    # test_nearest_neighbour()
    # test_rrt()
    # test_easy()
    test_hard()
