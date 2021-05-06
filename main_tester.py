import sys
import os
import shutil
import gensim
import random
import torch as torch
import torch.nn as nn
import torch.nn.functional as F
from time import time

from node_object_creator import *
from embeddings import Embedding
from node import Node
from matrix_generator import MatrixGenerator
from first_neural_network import First_neural_network
from coding_layer import Coding_layer
from convolutional_layer import Convolutional_layer
from pooling_layer import Pooling_layer
from dynamic_pooling import Max_pooling_layer, Dynamic_pooling_layer
from hidden_layer import Hidden_layer
from get_targets import GetTargets
from second_neural_network import SecondNeuralNetwork
from validation_neural_network import Validation_neural_network

    
#####################################
# SCRIPT
def main(vector_size, learning_rate, momentum, learning_rate2, feature_size, epoch, pooling, l2_penalty):


    ### Creation of the training set and validation set
    path = os.path.join('sets_short', 'generators')
    training_dict, validation_dict = training_and_validation_sets_creation(path) 


    
    ### Training set
    # this is the tensor with all target values
    targets = target_tensor_set_up(path, training_dict)

    # We now do the first neural network for every file:
    training_dict = first_neural_network(training_dict, vector_size, learning_rate, momentum, l2_penalty)

    # Training
    secnn = SecondNeuralNetwork(vector_size, feature_size, pooling)
    secnn.train(targets, training_dict, epoch, learning_rate2)

    # Validation
    val = Validation_neural_network(vector_size, feature_size, pooling)
    val.validation(path, validation_dict)


#####################################
# FUNCTIONS

def training_and_validation_sets_creation(path):
    # we create the training set and the validation set
    training_set = {}
    validation_set = {}
    # iterates through the generators directory, identifies the folders and enter in them
    for (dirpath, dirnames, filenames) in os.walk(path):
        for folder in dirnames:
            # we list all files of each folder
            folder_path = os.path.join(dirpath, folder)
            list_dir = os.listdir(folder_path)
            # Having a list with only .py files
            list_files_py = [file for file in list_dir if file.endswith('.py')]
            # we choose randomly 70% of this files
            # Number of files in the training set
            N = int(len(list_files_py)*0.7)
            i=1
            while list_files_py:
                file = random.choice(list_files_py)
                list_files_py.remove(file)
                if i <= N:
                    filepath = os.path.join(folder_path, file)
                    training_set[filepath] = None
                else:
                    filepath = os.path.join(folder_path, file)
                    validation_set[filepath] = None
                i += 1
    return training_set, validation_set


def target_tensor_set_up(path, training_dict):
    # Target dict initialization
    target = GetTargets(path)
    targets_dict = target.df_iterator()
    targets = []
    for filepath in training_dict.keys():
        # Targets' tensor creation
        split_filepath = os.path.split(filepath)
        filepath_target = 'label_' + split_filepath[1] + '.csv'
        search_target = os.path.join(split_filepath[0], filepath_target)
        if search_target in targets_dict.keys():
            if targets == []:
                targets = targets_dict[search_target]
            else:
                targets = torch.cat((targets, targets_dict[search_target]), 0)
    return targets


def first_neural_network(training_dict, vector_size = 20, learning_rate = 0.1, momentum = 0.01, l2_penalty = 0):
    total = len(training_dict)
    i = 1
    for data in training_dict:
        # Initializing node list, dict list and dict sibling

        # we parse the data of the file into a tree
        tree = file_parser(data)
        # convert its nodes into the Node class we have, and assign their attributes
        ls_nodes, dict_ast_to_Node = node_object_creator(tree)
        ls_nodes = node_position_assign(ls_nodes)
        ls_nodes, dict_sibling = node_sibling_assign(ls_nodes)
        ls_nodes = leaves_nodes_assign(ls_nodes, dict_ast_to_Node)

        # Initializing vector embeddings
        embed = Embedding(vector_size, ls_nodes, dict_ast_to_Node)
        ls_nodes = embed.node_embedding()

        # Calculate the vector representation for each node
        vector_representation = First_neural_network(ls_nodes, dict_ast_to_Node, vector_size, learning_rate, momentum, l2_penalty)
        ls_nodes, w_l_code, w_r_code, b_code = vector_representation.vector_representation()

        training_dict[data] = [ls_nodes, dict_ast_to_Node, dict_sibling, w_l_code, w_r_code, b_code]
        print(f"finished vector representation of file: {data} ({i}/{total})")
        i += 1
    return training_dict



########################################


if __name__ == '__main__':
    #first neural network parameters
    vector_size = 20
    learning_rate = 0.1
    momentum = 0.01
    l2_penalty = 0
    # Second neural network parameters
    learning_rate2 = 0.1
    feature_size = 4
    epoch = 10
    pooling = 'one-way pooling'

    main(vector_size, learning_rate, momentum, learning_rate2, feature_size, epoch, pooling, l2_penalty)


