from tree import Tree
import numpy as np


def get_random_state(ranges_starts, ranges):
    return np.random.rand(len(ranges_starts)) * ranges + ranges_starts


def move_car_like(start_state, steer_angle, car_length, time):
    phi = start_state[2] + time * np.tan(steer_angle) / car_length
    y = start_state[1] + time * np.sin(phi)
    x = start_state[0] + time * np.cos(phi)

    return np.array([x, y, phi])


def get_path(goal_node):
    path = []

    node = goal_node
    steer_angle = 0
    while node is not None:
        state = np.copy(node.state)
        tmp = state[3]
        state[3] = steer_angle # moving control inputs from children to parents
        steer_angle = tmp
        path.append(state)
        node = node.parent

    path.reverse()
    return np.array(path)


def solve_rrt_car_like(start, goal, goal_tolerance, boundaries, collision_detector,
                       steer_limit=np.pi / 3, n_actions=3, action_time=0.1, car_length=0.1, iters=1000):
    """
    Finds path in 2D space using car-like RRT algorithm with fixed speed.

    Args:
        start: (x, y, rotation) coordinates of the start position - numpy array (3, )
        goal: (x, y) coordinates of the goal position - numpy array (2, )
        goal_tolerance: defines the "goal region" around the goal position - float
        boundaries: (2, 2) numpy array containing lower (first) and upper (second) bounds for the x and y coordinates
                    in the 2D state space - e.g. boundaries = np.array([[-1, 1], [-2, 2]]) means x is from interval [-1, 1]
                    and y from [-2, 2]
        collision_detector: boolean predicate function which takes (x, y) coordinates ((2, ) numpy array) as input
                            and returns False if the state is permissible (collision free), True if the state collides
                            with obstacles
        steer_limit: defines maximal car steering angle - steering angle is then generated from the interval
                     [-steer_limit, steer_limit] - python float
        n_actions: number of actions (steering angles) tried during tree expansion
        action_time: time per one action ( = time between nodes on the path)
        car_length: length of the car-like model
        iters: number of iteration of the RRT algorithm

    Returns:
        path: sequence of states and actions: (x, y, rotation_angle, steer_angle) - numpy array (path_length, 4)
        tree: RRT tree - Tree object
    """
    steer_angles = [i * (2 * steer_limit / n_actions) - steer_limit for i in range(n_actions)]
    ranges = boundaries[:, 1] - boundaries[:, 0]  # used by get_random_state
    ranges_starts = boundaries[:, 0]

    tree = Tree()
    tree.add_node(start[:2], np.append(start, np.nan), None)  # NaN represent "missing" steering angle

    for i in range(iters):
        rand_coords = get_random_state(ranges_starts, ranges)
        near_node = tree.get_nearest(rand_coords)

        best = np.inf
        new_state = None
        for angle in steer_angles:
            state = move_car_like(near_node.state, angle, car_length, action_time)
            # TODO: add collision detection
            if np.linalg.norm(state[:2] - rand_coords) < best:
                new_state = np.append(state, angle) # storing steering angle with the new node (control input used in parent node)
                best = np.linalg.norm(state[:2] - rand_coords)

        if new_state is not None:
            new_node = tree.add_node(new_state[:2], new_state, near_node)
            if np.linalg.norm(new_state[:2] - goal) < goal_tolerance:
                # found solution
                return get_path(new_node), tree

    return None
