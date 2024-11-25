from shapely import Polygon, Point, LineString
from shapely.ops import nearest_points

from rrt_star import RRTStarPlanner
from car_like import solve_rrt_car_like
import numpy as np


class Env:
    """
    Simple 2D environment for a motion planning task with polygonal obstacles and point robot.
    """

    def __init__(self, width: float, height: float):
        """
        Args:
            width: width of the environment
            height: height of the environment
        """
        self.width = width
        self.height = height
        self.obstacles = []

    def size(self):
        """
        Returns width and height of the environment
        Returns:
        (width, height) tuple
        """
        return self.width, self.height

    def add_obstacle(self, boundary):
        """
        Adds polygonal obstacle to the environment

        Args:
            boundary: ordered sequence of (x, y) point tuples which represent the boundary of a polygon

        """
        self.obstacles.append(Polygon(boundary))

    def point_collides(self, coords):
        """
        Checks if the given coordinates are collision free (outside obstacles).

        Args:
            coords: a tuple (x, y) with the coordinates in the environment

        Returns:
            True if the point collides with obstacles in the environment, false otherwise.
        """

        point = Point(coords)
        for obstacle in self.obstacles:
            if obstacle.contains(point):
                return True

        return False

    def segment_collides(self, point1, point2):
        """
        Checks if the line segment between point1 and point2 is collision free (outside obstacles).

        Args:
            point1: a tuple (x, y) with the coordinates in the environment
            point2: a tuple (x, y) with the coordinates in the environment

        Returns:
            True if the line segment collides with obstacles in the environment, false otherwise.
        """

        segment = LineString([point1, point2])
        for obstacle in self.obstacles:
            if segment.intersects(obstacle):
                return True

        return False

    def nearest_free_point(self, coords):
        """
        Returns nearest free point to the query point.

        Args:
            coords: a tuple (x, y) with the coordinates of the query point

        Returns:
            Nearest free point to the query point - (x, y) tuple
        """
        point = Point(coords)
        for obstacle in self.obstacles:
            if obstacle.contains(point):
                nearest_point = nearest_points(obstacle.boundary, point)[0]
                return nearest_point.x, nearest_point.y

        return coords  # query point is outside obstacles

    def nearest_collision_point(self, coords):
        """
        Returns nearest collision point to the query point.

        Args:
            coords: a tuple (x, y) with the coordinates of the query point

        Returns:
            Nearest collision point to the query point - (x, y) tuple
        """
        if len(self.obstacles) < 1:
            return None

        point = Point(coords)
        nearest_obstacle = None
        min_distance = float('inf')

        for obstacle in self.obstacles:
            distance = point.distance(obstacle)
            if distance < min_distance:
                min_distance = distance
                nearest_obstacle = obstacle

        nearest_point = nearest_points(nearest_obstacle, point)[0]
        return nearest_point.x, nearest_point.y

    def run_RRTstar(self, start, goal, max_time):
        """
        Finds path from star to goal using the RRT* algorithm.

        Args:
            start: start coords - tuple (x, y)
            goal: goal coords - tuple (x, y)
            max_time: maximal time for the planning task (in seconds)

        Returns:
            Result path
        """
        planner = RRTStarPlanner(2, (0.0, self.width), lambda state: not self.point_collides((state[0], state[1])))
        planner.set_start_goal(np.array(start), np.array(goal))
        return planner.solve(max_time)

    def run_RRT_car_like(self, start, goal, goal_tolerance=0.05, max_iters=10000, steer_limit=np.pi / 4, n_actions=5,
                         action_time=0.05, car_length=0.1, n_collision_checks=3):
        """
        Finds path from start to goal using RRT car-like algorithm.

        Args:
            start: (x, y, rotation) coordinates of the start position
            goal: (x, y) coordinates of the goal position
            goal_tolerance: defines the "goal region" around the goal position - float
            steer_limit: defines maximal car steering angle - steering angle is then generated from the interval
                         [-steer_limit, steer_limit] - python float
            n_actions: number of actions (steering angles) tried during tree expansion
            action_time: time per one action ( = time between nodes on the path)
            car_length: length of the car-like model
            max_iters: maximal number of iteration of the RRT algorithm
            n_collision_checks: number of collision checks performed on one tree expansion

        Returns:
            (path, tree) tuple or None if path was not found
            path: sequence of states and actions: (x, y, rotation_angle, steer_angle) - numpy array (path_length, 4)
            tree: RRT tree - Tree object
        """
        boundaries = np.array([[0.0, self.width], [0.0, self.height]])
        return solve_rrt_car_like(np.array(start), np.array(goal), goal_tolerance, boundaries,
                                  lambda state: self.point_collides((state[0], state[1])),
                                  steer_limit, n_actions, action_time, car_length, max_iters, n_collision_checks)


# some sample Envs:

def load_hard_env():
    env = Env(1.0, 1.0)
    env.add_obstacle([(0.35, 0.11), (0.45, 0.22), (0.53, 0.06)])
    env.add_obstacle([(0.12, 0.30), (0.16, 0.48), (0.26, 0.37)])
    env.add_obstacle([(0.69, 0.22), (0.90, 0.40), (0.82, 0.16)])
    env.add_obstacle([(0.43, 0.42), (0.57, 0.33), (0.73, 0.42), (0.70, 0.58), (0.50, 0.60)])
    env.add_obstacle([(0.14, 0.65), (0.30, 0.60), (0.47, 0.75), (0.27, 0.71), (0.14, 0.80)])
    env.add_obstacle([(0.66, 0.71), (0.88, 0.73), (0.50, 0.90)])
    return env


def load_easy_env():
    env = Env(1.0, 1.0)
    env.add_obstacle([(0.32, 0.37), (0.29, 0.56), (0.42, 0.68), (0.58, 0.69), (0.70, 0.52), (0.66, 0.35), (0.49, 0.29)])
    return env
