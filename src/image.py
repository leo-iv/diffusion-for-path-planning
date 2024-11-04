import cairo
import numpy as np
import random

color_palette = [
    (12, 234, 217),
    (58, 202, 223),
    (114, 158, 253),
    (138, 100, 214),
    (92, 58, 146),
    (211, 253, 161),
    (150, 227, 165),
    (65, 190, 194),
    (253, 198, 117),
    (253, 157, 117)
]


class Image:
    """
    Renders the environment objects into a vector image.
    """

    def __init__(self, file_name, env, obstacle_color=(222, 196, 132), background_color=(255, 255, 255),
                 image_size=2000):
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
        # transforming polygon boundary into cairo coordinate system
        boundary = map(self.__transform_point, polygon.exterior.coords)

        x0, y0 = next(boundary)
        self.cr.move_to(x0, y0)
        for x, y in boundary:
            self.cr.line_to(x, y)

        self.cr.set_source_rgb(color[0] / 255, color[1] / 255, color[2] / 255)
        self.cr.fill()

    def add_circle(self, center, radius, color, transform=True):
        """
        Adds a circular point to the image.
        Args:
            center: center of the circle - (x, y) tuple
            radius: radius of the circle
            color: color of the circle - tuple (R, G, B) with the RGB color intensities (each from 0 to 255)
        """

        if transform:
            center = self.__transform_point(center)
        self.cr.set_source_rgb(color[0] / 255, color[1] / 255, color[2] / 255)
        self.cr.arc(center[0], center[1], radius, 0, 2 * np.pi)
        self.cr.fill()

    def add_star(self, center, radius, color):
        """
        Adds five-pointed star to the image.
        Args:
            center: center of the star - (x, y) tuple
            radius: radius of the star
            color: color of the star - tuple (R, G, B) with the RGB color intensities (each from 0 to 255)
        """
        num_points = 5
        center = self.__transform_point(center)
        self.cr.set_source_rgb(color[0] / 255, color[1] / 255, color[2] / 255)

        delta_angle = 2 * np.pi / (num_points * 2)
        start_angle = -np.pi / 2
        for i in range(num_points * 2):
            r = radius if i % 2 == 0 else radius / 2

            x = center[0] + r * np.cos(i * delta_angle + start_angle)
            y = center[1] + r * np.sin(i * delta_angle + start_angle)
            if i == 0:
                self.cr.move_to(x, y)
            else:
                self.cr.line_to(x, y)

        self.cr.fill()



    def add_path(self, path, color=(87, 126, 137), width=0.005):
        """
        Adds path to the image.

        Args:
            path: matrix where each row represent one point in the path - shape: (number of points in path, 2)
            color: color of the polygon - tuple (R, G, B) with the RGB color intensities (each from 0 to 255)
            width: width of the path
        """

        self.add_circle(path[0], 1.8 * width, color)
        for point in path[1:-1]:
            self.add_circle(point, 1.2 * width, color)
        self.add_star(path[-1], 3 * width, color)

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

    def add_paths(self, paths, use_one_color=False, color=(87, 126, 137), width=0.005):
        """
        Adds multiple paths to the image.

        Args:
            paths: numpy tensor with paths - shape: (number of paths, number of points in path, 2)
            use_one_color: set to True if all paths should have the same color - specified via the color param
            color: color of the paths (if use_one_color is set to True)
            width: width of the paths
        """
        for i, path in enumerate(paths):
            c = color
            if not use_one_color:
                if i < len(color_palette):
                    c = color_palette[i]
                else:
                    # generating random color instead
                    c = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            self.add_path(path, c, width)
