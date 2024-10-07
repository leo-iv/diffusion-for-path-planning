from env import Env


def test_rrt_star():
    env = Env(1.0, 1.0)
    env.add_obstacle([(0.35, 0.11), (0.45, 0.22), (0.53, 0.06)])
    env.add_obstacle([(0.12, 0.30), (0.16, 0.48), (0.26, 0.37)])
    env.add_obstacle([(0.69, 0.22), (0.90, 0.40), (0.82, 0.16)])
    env.add_obstacle([(0.43, 0.42), (0.57, 0.33), (0.73, 0.42), (0.70, 0.58), (0.50, 0.60)])
    env.add_obstacle([(0.14, 0.65), (0.30, 0.60), (0.47, 0.75), (0.27, 0.71), (0.14, 0.80)])
    env.add_obstacle([(0.66, 0.71), (0.88, 0.73), (0.50, 0.90)])

    paths = []
    time = 0.5
    paths.append(env.run_RRTstar((0.6, 0.1), (0.9, 0.9), time))
    paths.append(env.run_RRTstar((0.1, 0.1), (0.9, 0.6), time))
    paths.append(env.run_RRTstar((0.8, 0.1), (0.3, 0.9), time))
    paths.append(env.run_RRTstar((0.1, 0.4), (0.9, 0.5), time))
    paths.append(env.run_RRTstar((0.2, 0.1), (0.6, 0.9), time))

    img = env.create_image("../out/test_rrt_star.svg")
    for path in paths:
        img.add_path(path, (87, 126, 137), 0.005)


if __name__ == "__main__":
    test_rrt_star()
