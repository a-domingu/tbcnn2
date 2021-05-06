import numpy as np
import torch



class MatrixGenerator():
    ''' This class' purpose is to receive the list of nodes of a project as input, 
    and using their vector representations to generate a matrix using the idea behind 
    TBCNN, so that we can have the necessary input for a CNN.'''

    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.w = self.initalize_random_weight_matrix(self.n, self.m)
        self.b = self.initalize_random_bias_vector(self.n)


    def initalize_random_weight_matrix(self, n, m):
        # 'n' refers to the number of rows we want our matrix to have
        weight_matrix = torch.randn(n, m, requires_grad = True)
        return weight_matrix

    def initalize_random_bias_vector(self, n):
        bias = torch.randn(n, requires_grad = True)
        return bias








