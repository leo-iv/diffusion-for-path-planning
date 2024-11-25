from src.env import load_hard_env, load_easy_env
from tree import Tree
from env import Env
from image import Image
import numpy as np
from car_like import solve_rrt_car_like


def test_nearest_neighbour():
    env = Env(1.0, 1.0)
    img = Image("../out/car_like_test/nn_test.svg", env)
    tree = Tree()
    tree.add_node(np.array([0.1, 0.1]))
    tree.add_node(np.array([0.5, 0.5]))
    tree.add_node(np.array([0.9, 0.9]))
    tree.add_node(np.array([0.1, 0.9]))

    for node in tree.nodes:
        img.add_circle(node.coords, 0.01, (255, 0, 0))

    query = np.array([0.3, 0.8])
    nearest = tree.get_nearest(query)

    img.add_circle(query, 0.01, (0, 255, 0))
    img.add_star(nearest.coords, 0.03, (0, 0, 255))


def test_rrt():
    env = Env(1.0, 1.0)
    start = np.array([0.1, 0.1, 0])
    goal = np.array([0.9, 0.9])
    path, tree = env.run_RRT_car_like(start, goal, n_collision_checks=5, goal_bias=0.00001)
    img = Image("../out/car_like_test/rrt_test/empty.svg", env)
    img.add_tree_car_like(tree)
    img.add_path_car_like(path)

    env = load_hard_env()
    start = np.array([0.1, 0.1, 0])
    goal = np.array([0.9, 0.9])
    path, tree = env.run_RRT_car_like(start, goal, n_collision_checks=5)
    img = Image("../out/car_like_test/rrt_test/hard.svg", env)
    img.add_tree_car_like(tree)
    img.add_path_car_like(path)

    env = load_easy_env()
    start = np.array([0.1, 0.5, 0])
    goal = np.array([0.9, 0.5])
    path, tree = env.run_RRT_car_like(start, goal, n_collision_checks=5)
    img = Image("../out/car_like_test/rrt_test/easy.svg", env)
    img.add_tree_car_like(tree)
    img.add_path_car_like(path)




if __name__ == "__main__":
    # test_nearest_neighbour()
    test_rrt()
