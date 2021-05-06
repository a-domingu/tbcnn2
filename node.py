import ast
from relu import relu
import numpy as np
import torch as torch
import torch.nn as nn
import torch.nn.functional as F


class Node():
    '''
    For each node we store its parent and children nodes, as well as, its node type and its vector 
    representation
    '''
    def __init__(self, node, depth, parent = None):
        self.node = node
        self.children = self.get_children()
        self.parent = parent
        self.type = self.node.__class__.__name__
        self.vector = []
        self.combined_vector = []
        self.leaves_nodes = None
        self.depth = depth
        self.position = None
        self.siblings = None
        self.y = None
        self.pool = None

    def __str__(self):
        return self.type

    #Returns the children nodes of each node
    def get_children(self):
        ls = []
        for child in ast.iter_child_nodes(self.node):
            #nodeChild = Node(child, self)
            ls.append(child)
        return ls

    # Assigns the vector embedding to each node
    def set_vector(self, vector):
        if type(vector) == torch.Tensor:
            self.vector = vector
        else:
            self.vector = torch.tensor(vector, requires_grad = True)
    
    def set_combined_vector(self, vector):
        self.combined_vector = vector

    # Assigns the number of leaves nodes under each node
    def set_l(self, leaves_nodes):
        self.leaves_nodes = leaves_nodes

    def set_position(self, position):
        self.position = position
    
    def set_sibling(self, sibling):
        self.siblings = sibling

    def set_y(self, y):
        self.y = y

    def set_pool(self, pool):
        self.pool = pool