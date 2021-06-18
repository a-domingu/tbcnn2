import torch as torch
import torch.distributions
import torch.nn as nn
import torch.nn.functional as F

from first_neural_network.node import Node

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

    def __init__(self, vector_size, device, kernel_depth = 2, features_size = 4):
        self.vector_size = vector_size
        self.feature_size = features_size
        self.kernel_depth = kernel_depth
        self.device = device
        self.w_t = torch.distributions.Uniform(-1, +1).sample((self.feature_size, self.vector_size)).requires_grad_()
        #self.w_t = torch.rand(self.feature_size, self.vector_size).requires_grad_()
        self.w_r = torch.distributions.Uniform(-1, +1).sample((self.feature_size, self.vector_size)).requires_grad_()
        #self.w_r = torch.rand(self.feature_size, self.vector_size).requires_grad_()
        self.w_l = torch.distributions.Uniform(-1, +1).sample((self.feature_size, self.vector_size)).requires_grad_()
        #self.w_l = torch.rand(self.feature_size, self.vector_size).requires_grad_()
        self.b = torch.squeeze(torch.distributions.Uniform(-1, +1).sample((self.feature_size, 1))).requires_grad_()
        

    def convolutional_layer(self, ls_nodes):

        # self.y is the output of the convolutional layer.
        self.calculate_y(ls_nodes)

        return ls_nodes


    def initialize_matrices_and_bias(self):
        return self.w_t, self.w_l, self.w_r, self.b


    def set_matrices_and_vias(self, w_t, w_l, w_r, b):
        self.w_t, self.w_l, self.w_r, self.b = w_t, w_l, w_r, b


    def calculate_y(self, ls_nodes):
        for node in ls_nodes:
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
                vector_matrix, w_t_coeffs, w_l_coeffs, w_r_coeffs = node.matrix, node.coeff_t, node.coeff_l, node.coeff_r
                vector_matrix.to(self.device)
                w_t_coeffs.to(self.device)
                w_l_coeffs.to(self.device)
                w_r_coeffs.to(self.device)


                # The convolutional matrix for each node is a linear combination of matrices w_t, w_l and w_r
                convolutional_matrix = (w_t_coeffs*self.w_t) + (w_l_coeffs*self.w_l) + (w_r_coeffs*self.w_r)
                del w_t_coeffs
                del w_l_coeffs
                del w_r_coeffs

                final_matrix = torch.matmul(convolutional_matrix, vector_matrix)
                del vector_matrix
                final_vector = torch.sum(final_matrix, 0)
                final_vector = torch.squeeze(final_vector, 1)

                # When all the "weighted vectors" are added, we add on the b_conv.
                #argument = final_vector + self.b_conv

                # We used relu as the activation function in TBCNN mainly because we hope to 
                # encode features to a same semantic space during coding.
                node.set_y(F.leaky_relu(final_vector + self.b))
                del final_vector

            else:
                # The convolutional matrix for each node is a linear combination of matrices w_t, w_l and w_r
                #convolutional_matrix = self.w_t
                argument = torch.matmul(self.w_t, node.vector) + self.b
                node.set_y(F.leaky_relu(argument))

