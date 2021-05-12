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

    def __init__(self, n = 30, m = 100, pooling = 'one-way pooling', learning_rate2 = 0.01, epoch = 20):
        self.vector_size = n
        self.feature_size = m
        self.lr2 = learning_rate2
        self.epoch = epoch
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
        self.cod = Coding_layer(self.vector_size)
        self.conv = Convolutional_layer(self.vector_size, features_size=self.feature_size)
        self.hidden = Hidden_layer()


    def validation(self, targets, validation_dict):
        """Create the validation loop"""
        print('########################################')
        print('\n\n\nFinished training process. Entering validation process\n\n\n')
        print("The correct value of the files is: ", targets)


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
        print('Confusión matrix: ')
        print(conf_matrix)
        plot_confusion_matrix(conf_matrix, ['no generator', 'generator'], lr2 = self.lr2, feature_size = self.feature_size, epoch = self.epoch)


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


    def prediction(self, validation_dict):
        outputs = []
        softmax = nn.Sigmoid()
        for filepath in validation_dict:
            ## forward (second neural network)
            output = self.second_neural_network(validation_dict[filepath])

            # output append
            if outputs == []:
                outputs = torch.tensor([softmax(output)])
            else:
                outputs = torch.cat((outputs, torch.tensor([softmax(output)])), 0)

        return outputs


    def second_neural_network(self, vector_representation_params):
        ls_nodes = vector_representation_params[0]
        dict_ast_to_Node = vector_representation_params[1]
        dict_sibling = vector_representation_params[2]
        w_l_code = vector_representation_params[3]
        w_r_code = vector_representation_params[4]
        b_code = vector_representation_params[5]
        ls_nodes = self.cod.coding_layer(ls_nodes, dict_ast_to_Node, w_l_code, w_r_code, b_code, self.w_comb1, self.w_comb2)
        ls_nodes = self.conv.convolutional_layer(ls_nodes, dict_ast_to_Node, self.w_t, self.w_r, self.w_l, self.b_conv)
        if self.pooling == 'one-way pooling':
            vector = self.pooling_layer.pooling_layer(ls_nodes)
        else:
            self.max_pool.max_pooling(ls_nodes)
            vector = self.dynamic.three_way_pooling(ls_nodes, dict_sibling)
        output = self.hidden.hidden_layer(vector, self.w_hidden, self.b_hidden)

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