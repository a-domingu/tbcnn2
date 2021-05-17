import os
import random
import torch as torch
from time import time

from node_object_creator import *
from embeddings import Embedding
from first_neural_network import First_neural_network
from second_neural_network import SecondNeuralNetwork

    

def main(path, vector_size , learning_rate, momentum, l2_penalty, epoch_first, learning_rate2, feature_size, epoch, pooling, batch_size):
    # Training the first neural network
    data_dict = first_neural_network(path, vector_size, learning_rate, momentum, l2_penalty, epoch_first)

    # Training the second neural network
    second_neural_network(path, data_dict, vector_size, learning_rate2, feature_size, epoch, pooling, batch_size)


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
    for tree in data_dict:
        time1 = time()

        # convert its nodes into the Node class we have, and assign their attributes
        main_node = node_object_creator(tree)
    

        ls_nodes = main_node.descendants()
        set_leaves(ls_nodes)

        # Initializing vector embeddings
        embed = Embedding(vector_size, ls_nodes)
        embed.node_embedding()

        # Calculate the vector representation for each node
        vector_representation = First_neural_network(ls_nodes, vector_size, learning_rate, momentum, l2_penalty, epoch)
        ls_nodes, w_l_code, w_r_code, b_code = vector_representation.vector_representation()

        time2= time()
        dtime = time2 - time1

        data_dict[tree] = [ls_nodes, w_l_code, w_r_code, b_code]
        print(f"Vector rep. of file: {tree} ({i}/{total}) in ", dtime//60, 'min and', dtime%60, 'sec.')
        i += 1
    return data_dict


def second_neural_network(path, data_dict, vector_size, learning_rate2, feature_size, epoch, pooling, batch_size):
    ### Creation of the training set and validation set
    training_dict, validation_dict, targets_training, targets_validation = training_and_validation_sets_creation(path, data_dict) 

    # Training
    secnn = SecondNeuralNetwork(vector_size, feature_size, pooling)
    secnn.train(targets_training, training_dict, validation_dict, targets_validation, epoch, learning_rate2, batch_size)


def training_and_validation_sets_creation(path, data_dict):
    # we create the training set and the validation set
    training_set = {}
    validation_set = {}
    # We create a target training target tensor and a validation target tensor
    targets_training = [] 
    targets_validation = []
    # iterates through the generators directory, identifies the folders and enter in them
    for (dirpath, dirnames, filenames) in os.walk(path):
        for folder in dirnames:
            folder_path = os.path.join(dirpath, folder)
            if folder == 'withgen':
                training_set, validation_set, targets_training, targets_validation = tensor_creation(data_dict, folder_path, training_set, validation_set, targets_training, targets_validation, 1)
            elif folder == 'nogen':
                training_set, validation_set, targets_training, targets_validation = tensor_creation(data_dict, folder_path, training_set, validation_set, targets_training, targets_validation, 0)
            
    return training_set, validation_set, targets_training.float(), targets_validation.float()


def tensor_creation(data_dict, folder_path, training_set, validation_set, targets_training, targets_validation, value):
    # we list all files of each folder
    list_files = os.listdir(folder_path)
    # Having a list with only .py files
    list_files_py = [file for file in list_files if file.endswith('.py')]
    # we choose randomly 70% of this files
    # Number of files in the training set
    N = int(len(list_files_py)*0.7)
    i=1
    while list_files_py:
        file = random.choice(list_files_py)
        list_files_py.remove(file)
        if i <= N:
            filepath = os.path.join(folder_path, file)
            training_set[filepath] = data_dict[filepath]
            if targets_training == []:
                targets_training = torch.tensor([value])
            else:
                targets_training = torch.cat((targets_training, torch.tensor([value])), 0)
        else:
            filepath = os.path.join(folder_path, file)
            validation_set[filepath] = data_dict[filepath]
            if targets_validation == []:
                targets_validation = torch.tensor([value])
            else:
                targets_validation = torch.cat((targets_validation, torch.tensor([value])), 0)
        i += 1
    return training_set, validation_set, targets_training, targets_validation

def set_leaves(ls_nodes):
    for node in ls_nodes:
        node.set_leaves()


########################################

if __name__ == '__main__':
    # Folder path
    path = os.path.join('sets_short', 'generators')
    # First neural network parameters
    vector_size = 30
    learning_rate = 0.3
    momentum = 0
    l2_penalty = 0
    epoch_first = 1
    # Second neural network parameters
    learning_rate2 = 0.01
    feature_size = 50
    epoch = 2
    batch_size = 20
    pooling = 'one-way pooling'

    main(path, vector_size, learning_rate, momentum, l2_penalty, epoch_first, learning_rate2, feature_size, epoch, pooling, batch_size)