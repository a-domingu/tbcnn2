import os
import random
import torch as torch
import torch.nn as nn
from time import time
import pandas as pd
import pickle

from node_object_creator import *
from embeddings import Embedding
from first_neural_network import First_neural_network
from second_neural_network import SecondNeuralNetwork
from dataset import Dataset

    
def main(path, vector_size, learning_rate2, feature_size, epoch, pooling, batch_size):
        
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    params = {'batch_size': batch_size, 
    'shuffle': True, 
    'num_workers': 1} 


    ### Creation of the training set and validation set
    training_set, validation_set, training_targets, validation_targets = training_and_validation_sets_creation(path) 
    print('training set: ', training_set)
    print('targets del training: ', training_targets)

        
    # Generators
    training_dataset = Dataset(training_set, training_targets)
    training_generator = torch.utils.data.DataLoader(training_dataset, **params)

    validation_dataset = Dataset(validation_set, validation_targets)
    validation_generator = torch.utils.data.DataLoader(validation_dataset, **params)

    # Training
    model = SecondNeuralNetwork(device, vector_size, feature_size, pooling)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model.to(device)
    model.train(training_generator, validation_generator, epoch, learning_rate2, batch_size)


def training_and_validation_sets_creation(path):
    # we create the training set and the validation set
    training_set = []
    validation_set = []
    # We create a target training target tensor and a validation target tensor
    training_targets = {}
    validation_targets = {}
    #targets_training = [] 
    #targets_validation = []
    # iterates through the generators directory, identifies the folders and enter in them
    for (dirpath, dirnames, filenames) in os.walk(path):
        for folder in dirnames:
            folder_path = os.path.join(dirpath, folder)
            if folder == 'withgen':
                training_set, validation_set, training_targets, validation_targets = tensor_creation(folder_path, training_set, validation_set, training_targets, validation_targets, 1)
            elif folder == 'nogen':
                training_set, validation_set, training_targets, validation_targets = tensor_creation(folder_path, training_set, validation_set, training_targets, validation_targets, 0)
            
    return training_set, validation_set, training_targets, validation_targets


def tensor_creation(folder_path, training_set, validation_set, training_targets, validation_targets, value):
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
        filepath = os.path.join(folder_path, file)
        if i <= N:
            training_set.append(filepath)
            training_targets[filepath] = value
        else:
            validation_set.append(filepath)
            validation_targets[filepath] = value
        i += 1
    return training_set, validation_set, training_targets, validation_targets


########################################

if __name__ == '__main__':
    # Folder path
    path = os.path.join('sets_short', 'generators')
    # Second neural network parameters
    vector_size = 30
    learning_rate2 = 0.001
    feature_size = 50
    epoch = 5
    batch_size = 2
    pooling = 'one-way pooling'

    main(path, vector_size, learning_rate2, feature_size, epoch, pooling, batch_size)