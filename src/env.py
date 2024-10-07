from shapely import Polygon, Point
from image import Image
from src.rrt_star import RRTStarPlanner
import numpy as np


class Env:
    """
    Simple 2D environment for a motion planning task with polygonal obstacles and point robot.
    """
    __BACKGROUND_COLOR = (255, 255, 255)
    __OBSTACLE_COLOR = (222, 196, 132)
    __IMAGE_SIZE = 2000  # higher number means more pixels

    def __init__(self, width: float, height: float):
        """
        Args:
            width: width of the environment
            height: height of the environment
        """
        self.width = width
        self.height = height
        self.obstacles = []

    def add_obstacle(self, boundary):
        """
        Adds polygonal obstacle to the environment

        Args:
            boundary: ordered sequence of (x, y) point tuples which represent the boundary of a polygon

        """
        self.obstacles.append(Polygon(boundary))

    def check_collision(self, coords):
        """
        Checks if the given coordinates are collision free (outside obstacles).

        Args:
            coords: a tuple (x, y) with the coordinates in the environment

        Returns:
            True if the point is collision free, false otherwise.
        """

        for obstacle in self.obstacles:
            if obstacle.contains(Point(coords)):
                return False

        return True

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
        planner = RRTStarPlanner(2, (0.0, self.width), lambda state: self.check_collision((state[0], state[1])))
        planner.set_start_goal(np.array(start), np.array(goal))
        return planner.solve(max_time)

    def create_image(self, file_name: str):
        """
        Creates an SVG image of the environment and returns image.Image object for further modification of the image.

        Args:
            file_name: name of the SVG file (should end with .svg)

        Returns:
            image.Image object
        """
        img = Image(file_name, self.width, self.height, Env.__IMAGE_SIZE)
        img.fill(Env.__BACKGROUND_COLOR)
        for polygon in self.obstacles:
            img.add_polygon(polygon, Env.__OBSTACLE_COLOR)

        return img
