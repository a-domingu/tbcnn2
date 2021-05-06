import numpy as np
import random
import torch as torch
import torch.nn as nn
import torch.nn.functional as F

from node import Node
from matrix_generator import MatrixGenerator
from relu import relu

class Convolutional_layer():
    '''
    In this class we applied the tree-based convolution algorithm that we can find in section 4.2.4
    of the book "Tree-based Convolutional Neural Networks". Authors: Lili Mou and Zhi Jin
    We want to calculate the output of the feature detectors: vector y
    To do that, we have different elements an parameters:
    - Sliding window: In our case is a triangle that we use to extract the structural information of the AST.
        The sliding window has some features and parameters:
        - Kernel depth (or fixed-depth window): Number of hierarchical levels (or depths) inside of 
                                                the sliding window
        - d_i : Depth of node i in the sliding window. In our case the node at the top has the highest
                value, that corresponds with the value of the kernel depth; and the nodes at the bottom has
                the minimum value: 1.
        - d: Is the depth of the window, i.e, the kernel depth
        - p_i : Position of node i in the sliding window. In this case, is the position (1,..,N) 
                of the node in its hierarchical level (or depth) under the same parent within 
                the sliding window
        - n: Total number of siblings, i.e number of nodes on the same hierarchical level 
             under the same parent node within the sliding window
    - Feature detectors: Number of features that we want to study. It corresponds with the length of the 
                         output: vector y.
    Inputs:
    ls_nodes [list <class Node>]: list with all nodes in the AST
    dict_ast_to_Node[dict[ast_object] = <class Node>]: dictionary that relates class ast objects to class Node objects
    vector_size [int]: Vector embedding size
    kernel_depth [int]: Number of levels (or depths) in the sliding window
    features_size [int]: Number of feature detectors (N_c). Is the vector output size
    Output:
    ls_nodes [list <class Node>]: We add the output of feature detectors. It's the vector y
    w_t [matrix[features_size x vector_size]]: left weight matrix used as parameter
    w_r [matrix[features_size x vector_size]]: right weight matrix used as parameter
    w_l [matrix[features_size x vector_size]]: left weight matrix used as parameter
    b_conv [array[features_size]]: bias term
    '''

    def __init__(self, vector_size, w_t, w_r, w_l, b, kernel_depth = 2, features_size = 4):
        self.ls = []
        self.dict_ast_to_Node = {}
        self.vector_size = vector_size
        self.w_t = w_t
        self.w_r = w_r
        self.w_l = w_l
        self.b_conv = b
        self.Nc = features_size
        self.kernel_depth = kernel_depth


    def convolutional_layer(self, ls_nodes, dict_ast_to_Node):
        # Initialize the node list and the dict node
        self.ls = ls_nodes
        self.dict_ast_to_Node = dict_ast_to_Node

        # self.y is the output of the convolutional layer.
        self.calculate_y()

        return self.ls

    def calculate_y(self):

        for node in self.ls:
            if node.children:
                ''' We are going to create the sliding window. Taking as reference the book,
                we are going to set the kernel depth of our windows as 2. We consider into the window
                the node and its children.
                Question for ourselves: if we decide to increase the kernel depth to 3, should be
                appropiate to take node: its children and its grand-children or node, parent and children?
                '''
                # We create the sliding window with kernel depth = 2.
                window_nodes = [node]
                for child in node.children:  
                    window_nodes.append(self.dict_ast_to_Node[child])

                # We initialize the sum of weighted vector nodes
                sum = torch.zeros(self.Nc)

                '''
                We are going to calculate the parameters of the sliding window when its kernel depth
                is fixed to 2.
                In case we change the depth of the window, we have to change the parameters of this loop
                '''
                # We initialize p_i
                i = 1
                for item in window_nodes:
                    # We calculate the coefficients of the first node
                    if item == node:
                        # We set n = 2 because n cannot be 1 (eta_r uses n -1 as denominator)
                        n = 2
                        # Is the only node in the first level
                        p_i = 1
                        # Is the node at the top
                        d_i = 2
                    else:
                        # We calculate the number of node in the bottom level
                        n = len(node.children)
                        # If there is only one child, then we set n = 2 because n cannot be 1 
                        # (eta_r uses n -1 as denominator)
                        if n == 1:
                            n = 2
                        # We save the position of each node in the sliding window
                        p_i = i
                        i += 1
                        # The nodes are at the bottom
                        d_i = 1

                    # The weighted matrix for each node is a linear combination of matrices w_t, w_l and w_r
                    weighted_matrix = self.weight_matrix_update(d_i, self.kernel_depth, p_i, n)

                    sum = sum + torch.matmul(weighted_matrix,item.combined_vector)

                # When all the "weighted vectors" are added, we add on the b_conv.
                argument = sum + self.b_conv

                # We used relu as the activation function in TBCNN mainly because we hope to 
                # encode features to a same semantic space during coding.
                node.set_y(F.relu(argument))

            else:
                # The weighted matrix for each node is a linear combination of matrices w_t, w_l and w_r
                weighted_matrix = self.weight_matrix_update(self.kernel_depth, self.kernel_depth, 1, 2)
                argument = torch.matmul(weighted_matrix,node.combined_vector) + self.b_conv
                node.set_y(F.relu(argument))

    def weight_matrix_update(self, d_i, d, p_i, n):
        # The matrices coefficients are computed according to the relative position of 
        # a node in the sliding window.

        n_t = (d_i - 1)/(d-1)       # Coefficient associated to w_t
        n_r = (1-n_t)*(p_i-1)/(n-1) # Coefficient associated to w_r
        n_l = (1-n_t)*(1-n_r)        # Coefficient associated to w_l 

        '''
        print('AAAAAAAAAAAAa')
        print(type(n_t))
        print(self.w_t.size())
        '''

        top_matrix = n_t*self.w_t
        left_matrix = n_l* self.w_l
        right_matrix = n_r*self.w_r
        return (top_matrix + left_matrix + right_matrix) 