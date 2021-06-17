import torch 
import torch.nn as nn
import torch.nn.functional as F

from first_neural_network.node import Node

class Coding_layer():
    '''
    In this class we codify each node p as a combined vector of vec(·), where vec(·) 
    is the feature representation of a node in the AST.

    Inputs:
    ls_nodes [list <class Node>]: list with all nodes in the AST
    dict_ast_to_Node[dict[ast_object] = <class Node>]: dictionary that relates class ast objects to class Node objects
    features_size [int]: Vector embedding size
    w_l [matrix[features_size x features_size]]: left weight matrix used as parameter
    w_r [matrix[features_size x features_size]]: right weight matrix used as parameter
    b [array[features_size]]: bias term
    
    Output:
    ls_nodes [list <class Node>]: We update vector embedding of all nodes
    w_comb1 [matrix[features_size x features_size]]: Parameter 1 for combination
    w_comb2 [matrix[features_size x features_size]]: Parameter 2 for combination
    '''

    def __init__(self, features_size):
        self.ls = []
        self.dict_ast_to_Node = {}
        self.vector_size = features_size
        self.w_l = None
        self.w_r = None
        self.b = None
        self.w_comb1 = torch.diag(torch.squeeze(torch.distributions.Uniform(-1, +1).sample((self.vector_size, 1)), 1)).requires_grad_()
        self.w_comb2 = torch.diag(torch.squeeze(torch.distributions.Uniform(-1, +1).sample((self.vector_size, 1)), 1)).requires_grad_()


    def coding_layer(self, ls_nodes, w_l, w_r, b):
        # Initialize the node list and the dict node
        self.ls = ls_nodes
        # Initialize matrices and bias
        self.w_l = w_l
        self.w_r = w_r
        self.b = b

        self.coding_iterations()

        return self.ls


    def initialize_matrices_and_bias(self):
        return self.w_comb1, self.w_comb2


    def set_matrices_and_vias(self, w_comb1, w_comb2):
        self.w_comb1, self.w_comb2 = w_comb1, w_comb2


    # We create each combined vector p
    def coding_iterations(self):
        for node in self.ls:
            if len(node.children) > 1:
                combined_vector = self.node_coding(node)
                node.set_vector(combined_vector)
            elif len(node.children) == 1:
                combined_vector = self.node_coding_special_case(node)
                node.set_vector(combined_vector)
            else:
                combined_vector = torch.matmul(self.w_comb1, node.vector)
                node.set_vector(combined_vector)


    # Calculate the combination vector of each node p
    def node_coding(self, node):
        # Calculate the first term of the coding layers
        first_term = torch.matmul(self.w_comb1, node.vector)
        # Initalize the tensor of the second term
        sum = torch.zeros(self.vector_size, dtype=torch.float32)
        # Parameters used to calculate the weight matrix for each node
        n = len(node.children)
        i=1
        # number of leaves nodes under node p
        l_p = len(node.leaves)
        # Calculate the second term of the coding layer based on its child nodes
        for child in node.children:
            # The code matrix is weighted by the number of leaves nodes under child node
            matrix = ((len(child.leaves)/l_p))*self.weight_matrix(n, i)
            # Sum the weighted values over vec(child)
            sum = sum + torch.matmul(matrix, child.vector)
            i += 1
        children_part = F.leaky_relu(sum + self.b)
        second_term = torch.matmul(self.w_comb2, children_part)
        return (first_term + second_term)


    # Calculate the weighted matrix for each node as a linear combination of matrices w_l and w_r
    def weight_matrix(self, n, i):
        return (((n-i)/(n-1))*self.w_l) + (((i-1)/(n-1))*self.w_r)


    def node_coding_special_case(self, node):
        first_term = torch.matmul(self.w_comb1, node.vector)
        code_matrix = ((1/2)*self.w_l) + ((1/2)*self.w_r)
        matrix = (len(node.children[0].leaves)/len(node.leaves))*code_matrix
        second_term = torch.matmul(self.w_comb2, F.leaky_relu(torch.matmul(matrix, node.children[0].vector) + self.b))
        return (first_term + second_term)