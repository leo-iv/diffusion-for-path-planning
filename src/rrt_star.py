import numpy as np

import ompl.base as ob
import ompl.geometric as og


class RRTStarPlanner:
    """
    Wrapper class for the RRT* algorithm using the OMPL library.

    This class uses R^N (number of dimension specified in the constructor) as state space and assumes Euclidean geometry.
    """

    def __init__(self, N, bounds, collision_detector):
        """
        Args:
            N: number of dimension of the configuration space
            bounds: tuple (lower_bound, higher_bound) which contains bounds of the configuration space (same for each
                    dimension)
            collision_detector: predicate function which takes ompl.base.RealVectorStateSpace (state in the configuration
                                space) as input and returns true if the state is valid (doesn't collide with obstacles)
        """
        self.N = N
        self.space = ob.RealVectorStateSpace(N)

        bounds_obj = ob.RealVectorBounds(N)
        bounds_obj.setLow(bounds[0])
        bounds_obj.setHigh(bounds[1])

        self.space.setBounds(bounds_obj)

        self.ss = og.SimpleSetup(self.space)
        self.ss.setStateValidityChecker(ob.StateValidityCheckerFn(collision_detector))

        self.ss.setPlanner(og.RRTstar(self.ss.getSpaceInformation()))

    def __transform_path(self, path):
        """
        Transforms OMPL path to a np.array.
        """
        states = path.getStates()
        output = np.zeros((len(states), self.N))
        for i, state in enumerate(states):
            for dim in range(self.N):
                output[i, dim] = state[dim]

        return output

    def set_start_goal(self, start, goal):
        """
        Sets start and goal state for the motion planning task.

        Args:
            start: start configuration - np.array: shape: (n,)
            goal: goal configuration = np.array: shape: (n,)
        """
        start_state = ob.State(self.space)
        goal_state = ob.State(self.space)

        for i in range(self.N):
            start_state[i] = start[i]
            goal_state[i] = goal[i]

        self.ss.setStartAndGoalStates(start_state, goal_state)

    def solve(self, max_time):
        """
        Solves the motion planning task using RRT*. Can be called multiple times to obtain different paths. Previous call
        to set_start_goal() is expected.

        Args:
            max_time: float: maximal time for the planning task (in seconds)

        Returns:
            path from start to goal - numpy.ndarray matrix where each row contains one point in the path
        """
        self.ss.clear()
        self.ss.solve(max_time)
        return self.__transform_path(self.ss.getSolutionPath())
