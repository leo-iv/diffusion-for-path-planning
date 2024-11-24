import numpy as np


class Node:
    def __init__(self, coords, state, parent):
        self.coords = coords # coords used for nearest neighbour search
        self.state = state # state data
        self.children = []  # pointers to children nodes
        self.parent = parent  # pointer to parent node

    def add_child(self, child):
        self.children.append(child)


class Tree:
    """
    Tree graph implementation for the RRT algorithm.
    """

    def __init__(self):
        self.nodes = []

    def add_node(self, coords, state = None, parent: Node = None):
        """
        Adds one node to the graph (and edge from parent to the new node).
        The first added node is considered as the root of the tree (and should have None parent).

        Args:
            coords: coords in the RRT configuration space used for nearest neighbour search - (d, ) numpy array
            state: state data (if different from coords)
            parent: parent Node object

        Returns:
            new Node object
        """
        new_node = Node(coords, state, parent)
        if parent is not None:
            parent.add_child(new_node)
        self.nodes.append(new_node)
        return new_node

    def get_nearest(self, coords):
        """
        Returns the nearest Node to the query coords.

        Args:
            coords: (d, ) numpy array with query coords in the RRT configuration space

        Returns:
            Node object
        """
        # TODO: speed up using some kd-tree library
        min_node = None
        min_dist = np.inf

        for node in self.nodes:
            dist = np.linalg.norm(coords - node.coords)
            if dist < min_dist:
                min_node = node
                min_dist = dist

        return min_node

    def get_root(self):
        """
        Returns root Node of the Tree

        Returns:
            Node object first added to the tree.
        """
        if len(self.nodes) >= 1:
            return self.nodes[0]

        return None
