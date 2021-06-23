import ast
import numpy as np
import torch as torch
import torch.nn as nn
import torch.nn.functional as F

def get_descendants(node, ls):
    for child in node.children:
        ls.append(child)
        get_descendants(child, ls)
    return ls


class Node():
    '''
    For each node we store its parent and children nodes, as well as, its node type and its vector 
    representation
    '''
    def __init__(self, node, depth, parent = None):
        self.node = node
        self.children = []
        self.parent = parent
        self.type = self.get_node_type(node)
        self.vector = []
        self.combined_vector = []
        self.leaves = None
        self.depth = depth
        self.position = None
        self.siblings = None
        self.y = None
        self.pool = None


    def get_node_type(self, node):
        if node.__class__.__name__ == 'FunctionDef':
            if 'wrap' in node.name:
                return 'wrap'
            else:
                return node.__class__.__name__
        else:
            return node.__class__.__name__


    def __str__(self):
        return self.type

    #Returns the children nodes of each node
    def get_children(self):
        ls = []
        for child in ast.iter_child_nodes(self.node):
            #nodeChild = Node(child, self)
            ls.append(child)
        return ls

    def descendants(self):
        '''
        This function will return all of the nodes under the node itself.
        Note: the node itself is considered a descendant. This is done because it's useful when obtaining
        all of the nodes (otherwise we would miss the 'Module' node)
        '''
        ls = [self]
        return get_descendants(self, ls)

    # Assigns the vector embedding to each node
    def set_vector(self, df):
        if self.type in df.columns:
            vector =  df[self.type]
        else:
            num = len(df['Yield'])
            vector = [0 for i in range(num)]
        
        if type(vector) == torch.Tensor:
            self.vector = vector
        else:
            self.vector = torch.tensor(vector).float().requires_grad_()
    
    def set_combined_vector(self, vector):
        self.combined_vector = vector
    '''
    # Assigns the number of leaves nodes under each node
    def set_l(self, leaves_nodes):
        self.leaves_nodes = leaves_nodes
    '''
    def get_leaves(self):
    # TODO determinar cuándo hace falta hacer esto
        leaves = []
        descendants = self.descendants()
        for descendant in descendants:
            if descendant.children == []:
                leaves.append(descendant)
        return leaves

    def set_leaves(self):
        self.leaves = self.get_leaves()


    def set_y(self, y):
        self.y = y


    def set_pool(self, pool):
        self.pool = pool

    def set_children(self, child):
        self.children.append(child)

    def set_matrix_and_coeffs(self, matrix, t,l,r):
        self.matrix = matrix
        self.coeff_t = t
        self.coeff_l = l
        self.coeff_r = r