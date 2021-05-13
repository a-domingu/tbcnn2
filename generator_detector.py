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
from first_neural_network import First_neural_network
from coding_layer import Coding_layer
from convolutional_layer import Convolutional_layer
from pooling_layer import Pooling_layer
from dynamic_pooling import Max_pooling_layer, Dynamic_pooling_layer
from hidden_layer import Hidden_layer


class Generator_pattern_detection():

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
        self.cod = Coding_layer(self.vector_size)
        self.conv = Convolutional_layer(self.vector_size, features_size=self.feature_size)
        self.hidden = Hidden_layer()


    def generator_detection(self, path):
        """Create the data set"""
        print('########################################')
        print('Doing the embedding for each file \n')

        # Data dictionary creation
        data_dict = self.data_dict_creation(path)

        # Training the first neural network
        data_dict = self.first_neural_network(data_dict)

        # We calculate the predictions
        predicts = self.prediction(data_dict)
        
        # We print the predictions
        self.print_predictions(predicts, data_dict)


    def data_dict_creation(self, path):
        #TODO: implement a way to read also the files in a subfolder
        '''we want to read a path that can be a file or a folder with .py files'''
        # we create the data dict with all the information about vector representation
        data_dict = {}
        # iterates through the generators directory, identifies the folders and enter in them
        for (dirpath, dirnames, filenames) in os.walk(path):
            for filename in filenames:
                if filename.endswith('.py'):
                    filepath = os.path.join(dirpath, filename)
                    data_dict[filepath] = None


    def first_neural_network(data_dict, vector_size = 30, learning_rate = 0.3, momentum = 0, l2_penalty = 0, epoch = 45):
        total = len(data_dict)
        i = 1
        for data in data_dict:
            time1 = time()
            # Initializing node list, dict list and dict sibling

            # we parse the data of the file into a tree
            tree = file_parser(data)
            # convert its nodes into the Node class we have, and assign their attributes
            ls_nodes, dict_ast_to_Node = node_object_creator(tree)
            ls_nodes = node_position_assign(ls_nodes)
            ls_nodes, dict_sibling = node_sibling_assign(ls_nodes)
            ls_nodes = leaves_nodes_assign(ls_nodes, dict_ast_to_Node)

            # Initializing vector embeddings
            embed = Embedding(vector_size, ls_nodes, dict_ast_to_Node)
            ls_nodes = embed.node_embedding()

            # Calculate the vector representation for each node
            vector_representation = First_neural_network(ls_nodes, dict_ast_to_Node, vector_size, learning_rate, momentum, l2_penalty, epoch)
            ls_nodes, w_l_code, w_r_code, b_code = vector_representation.vector_representation()

            time2= time()
            dtime = time2 - time1

            data_dict[data] = [ls_nodes, dict_ast_to_Node, dict_sibling, w_l_code, w_r_code, b_code]
            print(f"Vector rep. of file: {data} ({i}/{total}) in ", dtime//60, 'min and', dtime%60, 'sec.')
            i += 1
        return data_dict


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


    def print_predictions(self, predicts, data_dict):
        i = 0
        for data in data_dict.keys():
            if predicts[i] < 0.5:
                print('The file ', data, ' has not generators')
            else:
                print('The file ', data, ' has generators')
            i+=1
