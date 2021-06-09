import os
import pandas as pd

from node_object_creator import *
from embeddings import Embedding

    

def main(pattern, vector_size, learning_rate = 0.3, momentum = 0, l2_penalty = 0, epoch_first = 1):
    # Training the first neural network
    vectors_dict = first_neural_network(pattern, vector_size, learning_rate, momentum, l2_penalty, epoch_first)

    #save_files(ls_nodes)
    save_vectors(vectors_dict)


def save_vectors(vectors_dict):
    df = pd.DataFrame.from_dict(vectors_dict)
    df.to_csv('initial_vector_representation.csv')


def first_neural_network(pattern, vector_size = 30, learning_rate = 0.3, momentum = 0, l2_penalty = 0, epoch = 1):
    # we create the data dict with all the information about vector representation
    data_dict = first_neural_network_dict_creation(pattern)
    # We now do the first neural network (vector representation) for every file:
    data_dict = vector_representation_all_files(data_dict, vector_size, learning_rate, momentum, l2_penalty, epoch)
    return data_dict


def first_neural_network_dict_creation(pattern):
    path = os.path.join('sets', pattern)
    # we create the data dict with all the information about vector representation
    data_dict = {}
    # iterates through the generators directory, identifies the folders and enter in them
    for (dirpath, _dirnames, filenames) in os.walk(path):
        if dirpath.endswith('withpattern') or dirpath.endswith('nopattern'):
            for filename in filenames:
                if filename.endswith('.py'):
                    filepath = os.path.join(dirpath, filename)
                    data_dict[filepath] = None

    return data_dict


def vector_representation_all_files(data_dict, vector_size = 20, learning_rate = 0.1, momentum = 0.01, l2_penalty = 0, epoch = 45):
    ls_nodes_all = []
    for tree in data_dict:
    
        # convert its nodes into the Node class we have, and assign their attributes
        main_node = node_object_creator(tree)
    
        for node in main_node.descendants():
            ls_nodes_all.append(node)

    # Initializing vector embeddings
    embed = Embedding(vector_size, ls_nodes_all)
    dc = embed.node_embedding()
    return dc


########################################

if __name__ == '__main__':
    # Folder path
    pattern = 'generators'
    # First neural network parameters
    vector_size = 30
    learning_rate = 0.3
    momentum = 0
    l2_penalty = 0
    epoch_first = 1

    main(pattern, vector_size, learning_rate, momentum, l2_penalty, epoch_first)