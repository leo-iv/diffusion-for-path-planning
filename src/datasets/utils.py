import random


def generate_start_goal(env):
    # generates free start and goal configurations in the environment at least half the env size apart
    width, height = env.size()

    def sample_start():
        # sample a random point from the "borders" of the env
        x = random.uniform(0, width / 2)
        y = random.uniform(0, height / 2)
        if x > width / 4:
            x += width / 2
        if y > height / 4:
            y += height / 2
        return x, y

    def sample_goal(start_coords):
        # sample from the opposite side of the env to start point
        start_x, start_y = start_coords
        if random.random() < 0.5:
            # "opposite x"
            if start_x < width / 4:
                x = random.uniform(3 * width / 4, width)
            else:
                x = random.uniform(0, width / 4)

            y = random.uniform(0, height)
        else:
            # "opposite y"
            if start_y < height / 4:
                y = random.uniform(3 * height / 4, height)
            else:
                y = random.uniform(0, height / 4)

            x = random.uniform(0, width)

        return x, y

    start = sample_start()
    while env.point_collides(start):
        start = sample_start()  # sample until we get collision free point

    stop = sample_goal(start)
    while env.point_collides(stop):
        stop = sample_goal(start)

    return start, stop
