import os
import sys
import random
import torch as torch
import torch.nn as nn
import importlib

from utils.node_object_creator import *
from second_neural_network.second_neural_network import SecondNeuralNetwork
from second_neural_network.generator_second_neural_network import Generator_second_neural_network
import second_neural_network.generator_second_neural_network
from second_neural_network.dataset import Dataset
from get_input import Get_input
from parameters import folder, pattern, vector_size, learning_rate2, feature_size, epoch, batch_size


class Pattern_training():

    def __init__(self, folder, pattern, vector_size, learning_rate2, feature_size, epoch, batch_size):
        self.folder = folder
        self.pattern = pattern
        self.vector_size = vector_size
        self.learning_rate = learning_rate2
        self.feature_size = feature_size
        self.epoch = epoch
        self.batch_size = batch_size
        self.pooling = "one_way pooling"

    
    def pattern_training(self):
            
        # CUDA for PyTorch
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
        torch.backends.cudnn.benchmark = True

        params = {'batch_size': self.batch_size, 
        'shuffle': True, 
        'num_workers': 8} 

        ### Creation of the training set and validation set
        training_set, validation_set, training_targets, validation_targets = self.training_and_validation_sets_creation() 
        print(f'We are using {device} device')
            
        # Dataset management: batch creation and sending data to GPU
        training_dataset = Dataset(training_set, training_targets)
        training_generator = torch.utils.data.DataLoader(training_dataset, **params)

        validation_dataset = Dataset(validation_set, validation_targets)
        validation_generator = torch.utils.data.DataLoader(validation_dataset, **params)

        # We instantiate the pattern class
        class_name = self.pattern.capitalize() + '_second_neural_network'
        module = importlib.import_module('second_neural_network.' + self.pattern + '_second_neural_network')
        pattern_class = getattr(module, class_name)

        # Training
        model = pattern_class(device, self.pattern, self.vector_size, self.feature_size, self.pooling)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            model = nn.DataParallel(model)

        model.to(device)
        model.train(training_generator, validation_generator, self.epoch, self.learning_rate, self.batch_size)


    def training_and_validation_sets_creation(self):
        path = os.path.join(self.folder, self.pattern)
        #If there is not a set with the required pattern, we print an error
        if not os.path.isdir(path):
            message = '''
            ---------------------------------------------------------------------------------
            This pattern is not implemented. Please check the following:
               - There is a labeled set for the required pattern.
               - There is a second neural network subclass implemented for this pattern.
               - The pattern name is well written.
            -----------------------------------------------------------------------------
            '''
            print(message)
            sys.exit()
        else:
            # we create the training set and the validation set
            training_set = []
            validation_set = []
            # We create a target training target tensor and a validation target tensor
            training_targets = {}
            validation_targets = {}
            # iterates through the generators directory, identifies the folders and enter in them
            for (dirpath, dirnames, filenames) in os.walk(path):
                for folder in dirnames:
                    folder_path = os.path.join(dirpath, folder)
                    if folder == 'withpattern':
                        training_set, validation_set, training_targets, validation_targets = self.tensor_creation(folder_path, training_set, validation_set, training_targets, validation_targets, 1)
                    elif folder == 'nopattern':
                        training_set, validation_set, training_targets, validation_targets = self.tensor_creation(folder_path, training_set, validation_set, training_targets, validation_targets, 0)
                    
            return training_set, validation_set, training_targets, validation_targets


    def tensor_creation(self, folder_path, training_set, validation_set, training_targets, validation_targets, value):
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

    pattern_training = Pattern_training(folder, pattern, vector_size, learning_rate2, feature_size, epoch, batch_size)
    pattern_training.pattern_training()