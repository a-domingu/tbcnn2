import sys
import os
import gensim
import random
import numpy
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
from utils import plot_confusion_matrix, writer


class Validation_neural_network():

    def __init__(self, n = 30, m = 100, pooling = 'one-way pooling'):
        self.vector_size = n
        self.feature_size = m
        # parameters
        w_comb1 = numpy.genfromtxt("params\\w_comb1.csv", delimiter = ",")
        self.w_comb1 = torch.tensor(w_comb1, dtype=torch.float32)
        w_comb2 = numpy.genfromtxt("params\\w_comb2.csv", delimiter = ",")
        self.w_comb2 = torch.tensor(w_comb2, dtype=torch.float32)
        w_t = numpy.genfromtxt("params\\w_t.csv", delimiter = ",")
        self.w_t = torch.tensor(w_t, dtype=torch.float32)
        w_r = numpy.genfromtxt("params\\w_r.csv", delimiter = ",")
        self.w_r = torch.tensor(w_r, dtype=torch.float32)
        w_l = numpy.genfromtxt("params\\w_l.csv", delimiter = ",")
        self.w_l = torch.tensor(w_l, dtype=torch.float32)
        b_conv = numpy.genfromtxt("params\\b_conv.csv", delimiter = ",")
        self.b_conv = torch.tensor(b_conv, dtype=torch.float32)
        w_hidden = numpy.genfromtxt("params\\w_hidden.csv", delimiter = ",")
        self.w_hidden = torch.tensor(w_hidden, dtype=torch.float32)
        b_hidden = numpy.genfromtxt("params\\b_hidden.csv", delimiter = ",")
        self.b_hidden = torch.tensor(b_hidden, dtype=torch.float32)

        # pooling method
        self.pooling = pooling
        if self.pooling == 'one-way pooling':
            self.pooling_layer = Pooling_layer()
        else:
            self.dynamic = Dynamic_pooling_layer()
            self.max_pool = Max_pooling_layer()

        ### Layers
        self.cod = Coding_layer(self.vector_size, self.w_comb1, self.w_comb2)
        self.conv = Convolutional_layer(self.vector_size, self.w_t, self.w_r, self.w_l, self.b_conv, features_size=self.feature_size)
        self.hidden = Hidden_layer(self.w_hidden, self.b_hidden)


    def validation(self, validation_path, validation_dict):
        """Create the validation loop"""
        print('########################################')
        print('\n\n\nFinished training process. Entering validation process\n\n\n')
        ### Validation set
        # this is to have all the information of each file in the folder contained in a dictionary
        #validation_dict = self.validation_dict_set_up(validation_path)
        # this is the tensor with all target values associated to the validation set
        targets = self.target_tensor_set_up(validation_path, validation_dict)


        # We calculate the predictions
        predicts = self.prediction(validation_dict)
        # print the predictions
        print('predictions: \n', predicts)

        # Loss function
        criterion = nn.BCELoss()
        loss = criterion(predicts, targets)

        # TODO Build the accuracy evaluation method for each file
        # Confusion matrix
        conf_matrix = self.conf_matrix(predicts, targets)
        print(conf_matrix)
        plot_confusion_matrix(conf_matrix, ['no generator', 'generator'])


        message = f'''

For the validation set we have the following results:
loss: {loss}
confusion_matrix:
{conf_matrix}
        '''
        writer(message)
        print('Loss validation: ', loss)
        # correct += (predicted == labels).sum()
        accuracy = self.accuracy(predicts, targets)
        print('accuracy: ', accuracy)



    def target_tensor_set_up(self, validation_path, validation_dict):
        # Target dict initialization
        target = GetTargets(validation_path)
        targets_dict = target.df_iterator()
        targets = []
        for filepath in validation_dict.keys():
            # Targets' tensor creation
            split_filepath = os.path.split(filepath)
            filepath_target = 'label_' + split_filepath[1] + '.csv'
            search_target = os.path.join(split_filepath[0], filepath_target)
            if search_target in targets_dict.keys():
                if targets == []:
                    targets = targets_dict[search_target]
                else:
                    targets = torch.cat((targets, targets_dict[search_target]), 0)
        print("The correct value of the files is: ", targets)
        return targets



    def prediction(self, validation_dict):
        outputs = []
        softmax = nn.Sigmoid()
        total = len(validation_dict)
        i = 1
        for filepath in validation_dict:
            # first neural network
            validation_dict[filepath] = self.first_neural_network(filepath)
            print(f"finished vector representation of file: {filepath} ({i}/{total}) \n")
            i += 1
            ## forward (second neural network)
            output = self.second_neural_network(validation_dict[filepath])

            # output append
            if outputs == []:
                outputs = torch.tensor([softmax(output)])
            else:
                outputs = torch.cat((outputs, torch.tensor([softmax(output)])), 0)

        return outputs
    

    def first_neural_network(self, file, learning_rate = 0.3, momentum = 0, l2_penalty = 0):
        '''Initializing node list, dict list and dict sibling'''
        # we parse the data of the file into a tree
        tree = file_parser(file)
        # convert its nodes into the Node class we have, and assign their attributes
        ls_nodes, dict_ast_to_Node = node_object_creator(tree)
        ls_nodes = node_position_assign(ls_nodes)
        ls_nodes, dict_sibling = node_sibling_assign(ls_nodes)
        ls_nodes = leaves_nodes_assign(ls_nodes, dict_ast_to_Node)

        # Initializing vector embeddings
        embed = Embedding(self.vector_size, ls_nodes, dict_ast_to_Node)
        ls_nodes = embed.node_embedding()

        # Calculate the vector representation for each node
        vector_representation = First_neural_network(ls_nodes, dict_ast_to_Node, self.vector_size, learning_rate, momentum, l2_penalty)
        ls_nodes, w_l_code, w_r_code, b_code = vector_representation.vector_representation()

        
        return [ls_nodes, dict_ast_to_Node, dict_sibling, w_l_code, w_r_code, b_code]


    def second_neural_network(self, vector_representation_params):
        ls_nodes = vector_representation_params[0]
        dict_ast_to_Node = vector_representation_params[1]
        dict_sibling = vector_representation_params[2]
        w_l_code = vector_representation_params[3]
        w_r_code = vector_representation_params[4]
        b_code = vector_representation_params[5]
        ls_nodes = self.cod.coding_layer(ls_nodes, dict_ast_to_Node, w_l_code, w_r_code, b_code)
        ls_nodes = self.conv.convolutional_layer(ls_nodes, dict_ast_to_Node)
        if self.pooling == 'one-way pooling':
            vector = self.pooling_layer.pooling_layer(ls_nodes)
        else:
            self.max_pool.max_pooling(ls_nodes)
            vector = self.dynamic.three_way_pooling(ls_nodes, dict_sibling)
        output = self.hidden.hidden_layer(vector)

        return output


    def accuracy(self, predicts, targets):
        with torch.no_grad():
            rounded_prediction = torch.round(predicts)

        # 1 if false negative
        # -1 if false positive
        difference = targets - rounded_prediction
        errors = torch.abs(difference).sum()

        accuracy = (len(difference) - errors)/len(difference)

        return accuracy
        
    def conf_matrix(self, predicts, targets):
        with torch.no_grad():
            rounded_prediction = torch.round(predicts)

        # 1 if false negative
        # -1 if false positive
        difference = targets - rounded_prediction

        # 0 if true negative
        # 2 if true positive
        addition = targets + rounded_prediction

        conf_matrix = torch.zeros(2,2, dtype=torch.int64)
        # x axis are true values, and y axis are predictions
        for i in range(len(addition)):
            if difference[i] == 1:
                conf_matrix[1,0] += 1
            elif difference[i] == -1:
                conf_matrix[0,1] += 1
            elif addition[i] == 0:
                conf_matrix[0,0] +=1
            else:
                assert addition[i] == 2
                conf_matrix[1,1] += 1
            
        return conf_matrix.numpy()