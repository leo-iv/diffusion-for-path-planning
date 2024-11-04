import shapely.ops
from shapely import Polygon, Point, LineString
from shapely.ops import nearest_points

from rrt_star import RRTStarPlanner
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
