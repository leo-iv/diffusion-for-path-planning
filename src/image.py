import cairo
import numpy as np


class Image:
    """
    Renders the environment objects into a vector image.
    """

    def __init__(self, file_name, env, obstacle_color = (222, 196, 132), background_color = (255, 255, 255), image_size = 2000):
        """
        Creates SVG image of the environment and returns Image class for further modification.

        Args:
            file_name: name of the file where the SVG image will be saved
            env: env.Env class instance
            image_size: width of the image (image height is calculated according to world height)
        """

        self.world_width, self.world_height = env.size()

        image_width = image_size
        image_height = image_size * (self.world_height / self.world_width)
        self.surface = cairo.SVGSurface(file_name, image_width, image_height)
        self.cr = cairo.Context(self.surface)
        self.cr.scale(image_width / self.world_width, image_height / self.world_height)

        # adding env picture
        self.fill(background_color)
        for polygon in env.obstacles:
            self.add_polygon(polygon, obstacle_color)


    def __transform_point(self, point):
        """
        Transforms a point from user (environment) coordinate system to cairo coordinate system
        Args:
            point: tuple (x, y)

        Returns:
            Transformed point - tuple (x, y)
        """
        return point[0], self.world_height - point[1]

    def fill(self, color):
        """
        Fills the image with color.

        Args:
            color: tuple (R, G, B) with the RGB color intensities (each from 0 to 255)
        """
        self.cr.set_source_rgb(color[0] / 255, color[1] / 255, color[2] / 255)
        self.cr.paint()

    def add_polygon(self, polygon, color):
        """
        Adds polygon to the image.

        Args:
            polygon: shapely.Polygon object
            color: color of the polygon - tuple (R, G, B) with the RGB color intensities (each from 0 to 255)
        """
        boundary = map(self.__transform_point,
                       polygon.exterior.coords)  # transforming polygon boundary into cairo coordinate system

        x0, y0 = next(boundary)
        self.cr.move_to(x0, y0)
        for x, y in boundary:
            self.cr.line_to(x, y)

        self.cr.set_source_rgb(color[0] / 255, color[1] / 255, color[2] / 255)
        self.cr.fill()

    def add_circle(self, center, radius, color, transform=True):
        """
        Adds a circular point to the image
        Args:
            center: center of the circle
            radius: radius of the circle
            color: color of the circle - tuple (R, G, B) with the RGB color intensities (each from 0 to 255)
        """

        if transform:
            center = self.__transform_point(center)
        self.cr.set_source_rgb(color[0] / 255, color[1] / 255, color[2] / 255)
        self.cr.arc(center[0], center[1], radius, 0, 2 * np.pi)
        self.cr.fill()

    def add_path(self, path, color = (87, 126, 137), width = 0.005):
        """
        Adds path to the image.

        Args:
            path: matrix where each row represent one point in the path - shape: (number of points in path, 2)
            color: color of the polygon - tuple (R, G, B) with the RGB color intensities (each from 0 to 255)
            width: width of the path
        """
        for point in path:
            self.add_circle(point, 1.2 * width, color)

        transformed_path = map(self.__transform_point, path)

        point_count, _ = path.shape

        self.cr.set_line_width(width)
        self.cr.set_source_rgb(color[0] / 255, color[1] / 255, color[2] / 255)

        x0, y0 = next(transformed_path)
        self.cr.move_to(x0, y0)
        for point in transformed_path:
            x, y = point
            self.cr.line_to(x, y)
        self.cr.stroke()
