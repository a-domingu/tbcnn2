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

    
def main(path, vector_size, learning_rate2, feature_size, epoch, pooling, batch_size):
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