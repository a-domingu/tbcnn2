import os
import pandas as pd
import torch as torch
import torch.nn as nn
from time import time
import pickle

from node_object_creator import *
from first_neural_network import First_neural_network
from coding_layer import Coding_layer
from convolutional_layer import Convolutional_layer
from dynamic_pooling import Dynamic_pooling_layer, Max_pooling_layer
from pooling_layer import Pooling_layer
from hidden_layer import Hidden_layer
from utils import conf_matrix, accuracy, bad_predicted_files


class Pattern_test():

    def __init__(self):
        self.vector_size = self.set_vector_size()


    def pattern_detection(self, pattern):

        # Load the trained matrices and vectors
        self.load_matrices_and_vectors()
        
        # Create the test set
        targets_set, targets_label = self.create_and_label_test_set(pattern)

        # Training the first neural network
        print('Doing the embedding for each file')
        print('######################################## \n')
        self.first_neural_network(targets_set)

        # We calculate the predictions
        predicts = self.prediction(targets_set, targets_label)
        self.delete_vector_representation_files()
        
        # We print the predictions
        self.print_predictions(targets_label, predicts, targets_set)

    
    def set_vector_size(self):
        df = pd.read_csv('initial_vector_representation.csv')
        vector_size = len(df[df.columns[0]])

        return vector_size

    
    def load_matrices_and_vectors(self):
        pass
        
    
    def create_and_label_test_set(self, pattern):
        path = os.path.join('test_sets', pattern)
        # We create a tensor with the name of the files and other tensor with their label
        targets_set = [] 
        targets_label = []
        # iterates through the generators directory, identifies the folders and enter in them
        for (dirpath, dirnames, filenames) in os.walk(path):
            if dirpath.endswith('withgen'):
                for filename in filenames:
                    if filename.endswith('.py'):
                        filepath = os.path.join(dirpath, filename)
                        targets_set.append(filepath)
                        targets_label.append(1)
            elif dirpath.endswith('nogen'):
                for filename in filenames:
                    if filename.endswith('.py'):
                        filepath = os.path.join(dirpath, filename)
                        targets_set.append(filepath)
                        targets_label.append(0)
        
        targets_label = torch.tensor(targets_label)
                
        return targets_set, targets_label


    def first_neural_network(self, targets_set, vector_size = 30, learning_rate = 0.3, momentum = 0, l2_penalty = 0, epoch = 1):
        i = 1
        for tree in targets_set:
            time1 = time()

            # convert its nodes into the Node class we have, and assign their attributes
            main_node = node_object_creator(tree)
            # we set the descendants of the main node and put them in a list
            ls_nodes = main_node.descendants()

            # We assign the leaves nodes under each node
            set_leaves(ls_nodes)
            # Initializing vector embeddings
            set_vector(ls_nodes)
            # Calculate the vector representation for each node
            vector_representation = First_neural_network(ls_nodes, vector_size, learning_rate, momentum, l2_penalty, epoch)
            ls_nodes, w_l_code, w_r_code, b_code = vector_representation.train()

            filename = os.path.join('vector_representation', os.path.basename(tree) + '.txt')
            params = [ls_nodes, w_l_code, w_r_code, b_code]

            with open(filename, 'wb') as f:
                pickle.dump(params, f)


            time2= time()
            dtime = time2 - time1

            print(f"Vector rep. of file: {tree} {i} in ", dtime//60, 'min and', dtime%60, 'sec.')
            i += 1


    def prediction(self, targets_set, targets_label):
        outputs = []
        softmax = nn.Sigmoid()
        for filepath in targets_set:
            filename = os.path.join('vector_representation', os.path.basename(filepath) + '.txt')
            with open(filename, 'rb') as f:
                params_first_neural_network = pickle.load(f)

            ## forward (second neural network)
            output = self.second_neural_network(params_first_neural_network)

            # output append
            if outputs == []:
                outputs = torch.tensor([softmax(output)])
            else:
                outputs = torch.cat((outputs, torch.tensor([softmax(output)])), 0)

        return outputs


    def second_neural_network(self, vector_representation_params):
        pass


    def delete_vector_representation_files(self):
        folder = 'vector_representation'
        for filename in os.listdir(folder):
            file = os.path.join(folder, filename)
            if file.startswith('.'):
                pass
            else:
                os.remove(file)


    def print_predictions(self, targets_label, predicts, targets_set):
        accuracy_value = accuracy(predicts, targets_label)
        print('Validation accuracy: ', accuracy_value)
        
        # Confusion matrix
        confusion_matrix = conf_matrix(predicts, targets_label)
        print('Confusión matrix: ')
        print(confusion_matrix)
        files_bad_predicted = bad_predicted_files(targets_set, predicts, targets_label)
        print(files_bad_predicted)


def set_leaves(ls_nodes):
    for node in ls_nodes:
        node.set_leaves()

def set_vector(ls_nodes):
    df = pd.read_csv('initial_vector_representation.csv')
    for node in ls_nodes:
        node.set_vector(df)
        
