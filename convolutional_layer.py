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
            ''' 
            We are going to create the sliding window. Taking as reference the book,
            we are going to set the kernel depth of our windows as 2. We consider into the window
            the node and its children.
            Question for ourselves: if we decide to increase the kernel depth to 3, should be
            appropiate to take node: its children and its grand-children or node, parent and children?

            We are going to calculate the parameters of the sliding window when its kernel depth
            is fixed to 2.
            In case we change the depth of the window, we have to change the parameters of each tensor
            '''
            if node.children:
                self.sliding_window_tensor(node)

                # The convolutional matrix for each node is a linear combination of matrices w_t, w_l and w_r
                convolutional_matrix = (self.w_t_params*self.w_t) + (self.w_l_params*self.w_l) + (self.w_r_params*self.w_r)

                final_matrix = torch.matmul(convolutional_matrix, self.vector_matrix)
                final_vector = torch.sum(final_matrix, 0)
                final_vector = torch.squeeze(final_vector, 1)

                # When all the "weighted vectors" are added, we add on the b_conv.
                argument = final_vector + self.b_conv

                # We used relu as the activation function in TBCNN mainly because we hope to 
                # encode features to a same semantic space during coding.
                node.set_y(F.relu(argument))

            else:
                # The convolutional matrix for each node is a linear combination of matrices w_t, w_l and w_r
                convolutional_matrix = self.w_t
                argument = torch.matmul(convolutional_matrix, node.combined_vector) + self.b_conv
                node.set_y(F.relu(argument))


    def sliding_window_tensor(self, node):
        # We create a list with all combined vectors
        vectors = [node.combined_vector]
        # Parameters used to calculate the convolutional matrix for each node
        n = len(node.children)
        # If there is only one child, then we set n = 2 because n cannot be 1 
        # (eta_r uses n -1 as denominator)
        if n == 1:
            n = 2
        d = self.kernel_depth
        # The nodes children are at the bottom
        d_i = 1
        # First node is the node at the top: d_i=2, p_i=1, n=2
        w_t_list = [(2-1)/(d-1)]
        w_r_list = [0]
        w_l_list = [0]
        i = 1
        for child in node.children:
            # We save the position of each node in the sliding window
            p_i = i
            w_t_list.append((d_i-1)/(d-1))
            w_r_list.append((1-w_t_list[i])*((p_i-1)/(n-1)))
            w_l_list.append((1-w_t_list[i])*(1-w_r_list[i]))
            i += 1
            # We save the combined vector of each node
            vectors.append(self.dict_ast_to_Node[child].combined_vector)

        # We create a matrix with all the vectors
        self.vector_matrix = torch.stack(tuple(vectors), 0)
        # We create a tensor with the parameters associated to the top matrix
        self.w_t_params = torch.tensor(w_t_list)
        # We create a tensor with the parameters associated to the left matrix
        self.w_l_params = torch.tensor(w_l_list)
        # We create a tensor with the parameters associated to the right matrix
        self.w_r_params = torch.tensor(w_r_list)
        # Reshape the matrices and vectors and create 3D tensors
        self.reshape_matrices_and_vectors()

    # Reshape the matrices and vectors and create 3D tensors
    def reshape_matrices_and_vectors(self):
        # We create a 3D tensor for the vector matrix: shape(nb_nodes, 30, 1)
        self.vector_matrix = torch.unsqueeze(self.vector_matrix, 2)

        # We create a 3D tensor for the parameters associated to the top matrix: shape(nb_nodes, 1, 1)
        self.w_t_params = torch.unsqueeze(self.w_t_params, 1)
        self.w_t_params = torch.unsqueeze(self.w_t_params, 1)

        # We create a 3D tensor for the parameters associated to the left matrix: shape(nb_nodes, 1, 1)
        self.w_l_params = torch.unsqueeze(self.w_l_params, 1)
        self.w_l_params = torch.unsqueeze(self.w_l_params, 1)

        # We create a 3D tensor for the parameters associated to the right matrix: shape(nb_nodes, 1, 1)
        self.w_r_params = torch.unsqueeze(self.w_r_params, 1)
        self.w_r_params = torch.unsqueeze(self.w_r_params, 1)