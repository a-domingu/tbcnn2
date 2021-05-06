import numpy as np
import random
import torch 
import torch.nn as nn
import torch.nn.functional as F

from node import Node
from matrix_generator import MatrixGenerator
from relu import relu

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

    def __init__(self, features_size, w1, w2):
        self.ls = []
        self.dict_ast_to_Node = {}
        self.features_size = features_size
        self.w_l = None
        self.w_r = None
        self.b = None
        self.w_comb1 = w1
        self.w_comb2 = w2


    def coding_layer(self, ls_nodes, dict_ast_to_Node, w_l, w_r, b):
        # Initialize the node list and the dict node
        self.ls = ls_nodes
        self.dict_ast_to_Node = dict_ast_to_Node
        # Initialize matrices and bias
        self.w_l = w_l
        self.w_r = w_r
        self.b = b

        self.coding_iterations()

        return self.ls


    # We create each combined vector p
    def coding_iterations(self):
        for node in self.ls:
            if len(node.children) > 1:
                combined_vector = self.node_coding(node)
                #print(combined_vector)
                node.set_combined_vector(combined_vector)
            elif len(node.children) == 1:
                combined_vector = self.node_coding_special_case(node)
                #print(combined_vector)
                node.set_combined_vector(combined_vector)
            else:
                combined_vector = torch.matmul(self.w_comb1, node.vector)
                node.set_combined_vector(combined_vector)


    # Calculate the combination vector of each node p
    def node_coding(self, node):
        # Calculate the first term of the coding layers
        first_term = torch.matmul(self.w_comb1, node.vector)
        # Initalize the tensor of the second term
        sum = torch.zeros(self.features_size, dtype=torch.float32)
        # Parameters used to calculate the weight matrix for each node
        n = len(node.children)
        i=1
        # number of leaves nodes under node p
        l_p = node.leaves_nodes
        # Calculate the second term of the coding layer based on its child nodes
        for child in node.children:
            # We convert the AST object to a Node object
            child_node = self.dict_ast_to_Node[child] 
            # number of leaves nodes under child node
            l_c = child_node.leaves_nodes
            l = (l_c/l_p)
            # Calculate the code matrix
            code_matrix = self.weight_matrix(n, i)
            # The code matrix is weighted by the number of leaves nodes under child node
            matrix = l*code_matrix
            # Sum the weighted values over vec(child)
            sum = sum + torch.matmul(matrix, child_node.vector)
            i += 1
        children_part = F.relu(sum + self.b)
        second_term = torch.matmul(self.w_comb2, children_part)
        return (first_term + second_term)


    # Calculate the weighted matrix for each node as a linear combination of matrices w_l and w_r
    def weight_matrix(self, n, i):
        left_matrix = ((n-i)/(n-1))* self.w_l
        right_matrix = ((i-1)/(n-1))*self.w_r
        return (left_matrix + right_matrix) 


    def node_coding_special_case(self, node):
        first_term = torch.matmul(self.w_comb1, node.vector)
        code_matrix = ((1/2)*self.w_l) + ((1/2)*self.w_r)
        matrix = (self.dict_ast_to_Node[node.children[0]].leaves_nodes/node.leaves_nodes)*code_matrix
        second_term = torch.matmul(self.w_comb2, F.relu(torch.matmul(matrix, self.dict_ast_to_Node[node.children[0]].vector) + self.b))
        return (first_term + second_term)