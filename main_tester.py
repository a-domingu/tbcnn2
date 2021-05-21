import os
import random
import torch as torch
from time import time
import pandas as pd
import pickle

from node_object_creator import *
from embeddings import Embedding
from first_neural_network import First_neural_network
from second_neural_network import SecondNeuralNetwork

    

def main(path, vector_size , learning_rate, momentum, l2_penalty, epoch_first, learning_rate2, feature_size, epoch, pooling, batch_size):
    # Training the first neural network
    first_neural_network(path, vector_size, learning_rate, momentum, l2_penalty, epoch_first)

    #save_files(data_dict)
    # Training the second neural network
    second_neural_network(path, vector_size, learning_rate2, feature_size, epoch, pooling, batch_size)

def save_files(dc):
    path = os.path.join('yield_results', 'yield.txt')
    with open('yield.txt', 'w') as f:
        for file in dc:
            ls_nodes = dc[file][0]
            f.write('####################################\n\n ')
            f.write(file)
            for node in ls_nodes:
                str_to_save = f'Node type: {node.type} ------- vector: {node.vector}\n'
                f.write(str_to_save)


def read_folder_data_set(path):
    # iterates through the generators directory, identifies the folders and enter in them
    for (dirpath, _dirnames, filenames) in os.walk(path):
        if dirpath.endswith('withgen') or dirpath.endswith('nogen'):
            for filename in filenames:
                if filename.endswith('.py'):
                    filepath = os.path.join(dirpath, filename)
                    yield filepath



def first_neural_network(path, vector_size = 20, learning_rate = 0.1, momentum = 0.01, l2_penalty = 0, epoch = 45):
    i = 1
    for tree in read_folder_data_set(path):
        time1 = time()

        # convert its nodes into the Node class we have, and assign their attributes
        main_node = node_object_creator(tree)
        ls_nodes = main_node.descendants()
        del main_node

        set_leaves(ls_nodes)
        # Initializing vector embeddings
        set_vector(ls_nodes)
        '''
        print('Initial vectors: ')

        for node in ls_nodes:
            print(node.vector)
            break

        for node in main_node.descendants():
            print(node.vector)
            print('####################')
            break
        '''
        # Calculate the vector representation for each node
        vector_representation = First_neural_network(ls_nodes, vector_size, learning_rate, momentum, l2_penalty, epoch)
        ls_nodes, w_l_code, w_r_code, b_code = vector_representation.vector_representation()
        '''
        print('After first neural network: ')

        for node in ls_nodes:
            print(node.vector)
            break

        for node in main_node.descendants():
            print(node.vector)
            print('####################')
            break
        '''
        filename = os.path.join('vector_representation', os.path.basename(tree) + '.txt')
        params = [ls_nodes, w_l_code, w_r_code, b_code]
        del ls_nodes
        del  w_l_code
        del w_r_code
        del b_code

        with open(filename, 'wb') as f:
            pickle.dump(params, f)
        
        del params

        time2= time()
        dtime = time2 - time1
        print(f"Vector rep. of file: {tree} {i} in ", dtime//60, 'min and', dtime%60, 'sec.')
        '''
        with open(filename, 'rb') as f:
            params_2 = pickle.load(f) 

        ls_nodes_2 = params_2[0]

        print('After save vectors with pickle: ')
        
        for node in ls_nodes_2:
            print(node.vector)
            break

        main_node_2 = ls_nodes_2[0]
        for node in main_node_2.descendants():
            print(node.vector)
            print('#################### \n')
            break
        '''
        i += 1


def second_neural_network(path, vector_size, learning_rate2, feature_size, epoch, pooling, batch_size):
    ### Creation of the training set and validation set
    training_set, validation_set, targets_training, targets_validation = training_and_validation_sets_creation(path) 

    # Training
    secnn = SecondNeuralNetwork(vector_size, feature_size, pooling)
    secnn.train(targets_training, training_set, validation_set, targets_validation, epoch, learning_rate2, batch_size)


def training_and_validation_sets_creation(path):
    # we create the training set and the validation set
    training_set = []
    validation_set = []
    # We create a target training target tensor and a validation target tensor
    targets_training = [] 
    targets_validation = []
    # iterates through the generators directory, identifies the folders and enter in them
    for (dirpath, dirnames, filenames) in os.walk(path):
        for folder in dirnames:
            folder_path = os.path.join(dirpath, folder)
            if folder == 'withgen':
                training_set, validation_set, targets_training, targets_validation = tensor_creation(folder_path, training_set, validation_set, targets_training, targets_validation, 1)
            elif folder == 'nogen':
                training_set, validation_set, targets_training, targets_validation = tensor_creation(folder_path, training_set, validation_set, targets_training, targets_validation, 0)
            
    return training_set, validation_set, targets_training.float(), targets_validation.float()


def tensor_creation(folder_path, training_set, validation_set, targets_training, targets_validation, value):
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
            training_set.append(filepath)
            if targets_training == []:
                targets_training = torch.tensor([value])
            else:
                targets_training = torch.cat((targets_training, torch.tensor([value])), 0)
        else:
            filepath = os.path.join(folder_path, file)
            validation_set.append(filepath)
            if targets_validation == []:
                targets_validation = torch.tensor([value])
            else:
                targets_validation = torch.cat((targets_validation, torch.tensor([value])), 0)
        i += 1
    return training_set, validation_set, targets_training, targets_validation

def set_leaves(ls_nodes):
    for node in ls_nodes:
        node.set_leaves()

def set_vector(ls_nodes):
    df = pd.read_csv('initial_vector_representation.csv')
    for node in ls_nodes:
        node.set_vector(df)


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
    learning_rate2 = 0.001
    feature_size = 50
    epoch = 5
    batch_size = 20
    pooling = 'one-way pooling'

    main(path, vector_size, learning_rate, momentum, l2_penalty, epoch_first, learning_rate2, feature_size, epoch, pooling, batch_size)