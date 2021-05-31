import os
import random
import torch as torch
from time import time
import pandas as pd

from node_object_creator import *
from embeddings import Embedding
from first_neural_network import First_neural_network
from second_neural_network import SecondNeuralNetwork

    

def main(path, vector_size , learning_rate, momentum, l2_penalty, epoch_first, learning_rate2, feature_size, epoch, pooling):
    # Training the first neural network
    vectors_dict = first_neural_network(path, vector_size, learning_rate, momentum, l2_penalty, epoch_first)

    #save_files(ls_nodes)
    save_vectors(vectors_dict)

    # Training the second neural network


def save_vectors(vectors_dict):
    df = pd.DataFrame.from_dict(vectors_dict)
    df.to_csv('initial_vector_representation.csv')

def first_neural_network(path, vector_size = 20, learning_rate = 0.1, momentum = 0.01, l2_penalty = 0, epoch = 45):
    # we create the data dict with all the information about vector representation
    data_dict = first_neural_network_dict_creation(path)
    # We now do the first neural network (vector representation) for every file:
    data_dict = vector_representation_all_files(data_dict, vector_size, learning_rate, momentum, l2_penalty, epoch)
    return data_dict


def first_neural_network_dict_creation(path):
    # we create the data dict with all the information about vector representation
    data_dict = {}
    # iterates through the generators directory, identifies the folders and enter in them
    for (dirpath, _dirnames, filenames) in os.walk(path):
        if dirpath.endswith('withgen') or dirpath.endswith('nogen'):
            for filename in filenames:
                if filename.endswith('.py'):
                    filepath = os.path.join(dirpath, filename)
                    data_dict[filepath] = None

    return data_dict


def vector_representation_all_files(data_dict, vector_size = 20, learning_rate = 0.1, momentum = 0.01, l2_penalty = 0, epoch = 45):
    total = len(data_dict)
    i = 1
    ls_nodes_all = []
    for tree in data_dict:
        time1 = time()

        # convert its nodes into the Node class we have, and assign their attributes
        main_node = node_object_creator(tree)
    
        for node in main_node.descendants():
            ls_nodes_all.append(node)

    # Initializing vector embeddings
    for node in ls_nodes_all:
        print(node)
        if node.__class__.__name__ == 'Str':
            print('AAAAAAAAAAAAAAAAAAAA')
            print(node)
            break
    raise Exception
    embed = Embedding(vector_size, ls_nodes_all)
    dc = embed.node_embedding()
    return dc

'''
    # Calculate the vector representation for each node
    vector_representation = First_neural_network(ls_nodes_all, vector_size, learning_rate, momentum, l2_penalty, epoch)

    ls_nodes, w_l_code, w_r_code, b_code = vector_representation.vector_representation()

    time2= time()
    dtime = time2 - time1

    #data_dict[tree] = [ls_nodes, w_l_code, w_r_code, b_code

    return ls_nodes
'''




########################################

if __name__ == '__main__':
    # Folder path
    path = os.path.join('sets', 'generators')
    # First neural network parameters
    vector_size = 30
    learning_rate = 0.3
    momentum = 0
    l2_penalty = 0
    epoch_first = 1
    # Second neural network parameters
    learning_rate2 = 0.01
    feature_size = 100
    epoch = 2
    pooling = 'one-way pooling'

    main(path, vector_size, learning_rate, momentum, l2_penalty, epoch_first, learning_rate2, feature_size, epoch, pooling)