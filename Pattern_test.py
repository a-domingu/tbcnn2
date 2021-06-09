import os
import numpy
import pandas as pd
import torch as torch
import torch.nn as nn
from time import time
import pickle
import shutil

from node_object_creator import *
from first_neural_network import First_neural_network
from coding_layer import Coding_layer
from convolutional_layer import Convolutional_layer
from dynamic_pooling import Dynamic_pooling_layer, Max_pooling_layer
from pooling_layer import Pooling_layer
from hidden_layer import Hidden_layer
from utils import conf_matrix, accuracy, bad_predicted_files


class Pattern_test():

    def __init__(self, n = 30, m = 100, pooling = 'one-way pooling'):
        self.vector_size = n
        self.feature_size = m
        # parameters
        w_comb1 = numpy.genfromtxt(os.path.join("params","w_comb1.csv"), delimiter = ",")
        self.w_comb1 = torch.tensor(w_comb1, dtype=torch.float32)
        w_comb2 = numpy.genfromtxt(os.path.join("params","w_comb2.csv"), delimiter = ",")
        self.w_comb2 = torch.tensor(w_comb2, dtype=torch.float32)
        w_t = numpy.genfromtxt(os.path.join("params","w_t.csv"), delimiter = ",")
        self.w_t = torch.tensor(w_t, dtype=torch.float32)
        w_r = numpy.genfromtxt(os.path.join("params","w_r.csv"), delimiter = ",")
        self.w_r = torch.tensor(w_r, dtype=torch.float32)
        w_l = numpy.genfromtxt(os.path.join("params","w_l.csv"), delimiter = ",")
        self.w_l = torch.tensor(w_l, dtype=torch.float32)
        b_conv = numpy.genfromtxt(os.path.join("params","b_conv.csv"), delimiter = ",")
        self.b_conv = torch.tensor(b_conv, dtype=torch.float32)
        w_hidden = numpy.genfromtxt(os.path.join("params","w_hidden.csv"), delimiter = ",")
        self.w_hidden = torch.tensor(w_hidden, dtype=torch.float32)
        b_hidden = numpy.genfromtxt(os.path.join("params","b_hidden.csv"), delimiter = ",")
        self.b_hidden = torch.tensor(b_hidden, dtype=torch.float32)

        # pooling method
        self.pooling = pooling
        if self.pooling == 'one-way pooling':
            self.pooling_layer = Pooling_layer()
        else:
            self.dynamic = Dynamic_pooling_layer()
            self.max_pool = Max_pooling_layer()

        ### Layers
        self.cod = Coding_layer(self.vector_size)
        self.conv = Convolutional_layer(self.vector_size, features_size=self.feature_size)
        self.hidden = Hidden_layer()


    def pattern_detection(self, path):
        
        """Create the test set"""
        targets_set, targets_label = self.create_and_label_test_set(path)

        # Training the first neural network
        print('Doing the embedding for each file')
        print('######################################## \n')
        self.first_neural_network(targets_set)

        # We calculate the predictions
        predicts = self.prediction(targets_set, targets_label)
        shutil.rmtree('vector_representation', ignore_errors=True)
        
        # We print the predictions
        self.print_predictions(targets_label, predicts, targets_set)

    
    def create_and_label_test_set(self, path):
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
            ls_nodes, w_l_code, w_r_code, b_code = vector_representation.vector_representation()

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
        ls_nodes, w_l_code, w_r_code, b_code = vector_representation_params
        # we don't do coding layer and go directly to convolutional layer; that's why we don't
        # use the matrices above
        ls_nodes = self.conv.convolutional_layer(ls_nodes, self.w_t, self.w_r, self.w_l, self.b_conv)
        if self.pooling == 'one-way pooling':
            vector = self.pooling_layer.pooling_layer(ls_nodes)
        else:
            self.max_pool.max_pooling(ls_nodes)
            dict_sibling = None
            vector = self.dynamic.three_way_pooling(ls_nodes, dict_sibling)
        output = self.hidden.hidden_layer(vector, self.w_hidden, self.b_hidden)

        return output


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
        



if __name__ == '__main__':
    path = os.path.join('sets', 'generators')

    generator_test = Pattern_test()
    generator_test.pattern_detection(path)
