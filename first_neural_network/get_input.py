import torch as torch
import torch.distributions
import torch.nn as nn
import torch.nn.functional as F
import pickle
import os

from first_neural_network.node import Node

class Get_input():

    def __init__(self, ls_nodes, vector_size, kernel_depth = 2, features_size = 4):
        self.vector_size = vector_size
        self.feature_size = features_size
        self.kernel_depth = kernel_depth
        self.ls_nodes = ls_nodes


    def get_input(self):
        for node in self.ls_nodes:
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
                vector_matrix, w_t_coeffs, w_l_coeffs, w_r_coeffs = self.sliding_window_tensor(node)
                node.set_matrix_and_coeffs(vector_matrix, w_t_coeffs, w_l_coeffs, w_r_coeffs)



    def sliding_window_tensor(self, node):
        # We create a list with all combined vectors
        vectors = [node.vector]
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
            vectors.append(child.vector)

        # We create a matrix with all the vectors
        vector_matrix = torch.stack(tuple(vectors), 0)
        del vectors
        # We create a tensor with the parameters associated to the top matrix
        w_t_params = torch.tensor(w_t_list)
        del w_t_list
        # We create a tensor with the parameters associated to the left matrix
        w_l_params = torch.tensor(w_l_list)
        del w_l_list
        # We create a tensor with the parameters associated to the right matrix
        w_r_params = torch.tensor(w_r_list)
        del w_r_list
        # Reshape the matrices and vectors and create 3D tensors
        vector_matrix, w_t_params, w_l_params, w_r_params = self.reshape_matrices_and_vectors(vector_matrix, w_t_params, w_l_params, w_r_params)
        return vector_matrix, w_t_params, w_l_params, w_r_params

    # Reshape the matrices and vectors and create 3D tensors
    def reshape_matrices_and_vectors(self, vector_matrix, w_t_params, w_l_params, w_r_params):
        # We create a 3D tensor for the vector matrix: shape(nb_nodes, 30, 1)
        vector_matrix = torch.unsqueeze(vector_matrix, 2)

        # We create a 3D tensor for the parameters associated to the top matrix: shape(nb_nodes, 1, 1)
        w_t_params = torch.unsqueeze(w_t_params, 1)
        w_t_params = torch.unsqueeze(w_t_params, 1)

        # We create a 3D tensor for the parameters associated to the left matrix: shape(nb_nodes, 1, 1)
        w_l_params = torch.unsqueeze(w_l_params, 1)
        w_l_params = torch.unsqueeze(w_l_params, 1)

        # We create a 3D tensor for the parameters associated to the right matrix: shape(nb_nodes, 1, 1)
        w_r_params = torch.unsqueeze(w_r_params, 1)
        w_r_params = torch.unsqueeze(w_r_params, 1)

        return vector_matrix, w_t_params, w_l_params, w_r_params