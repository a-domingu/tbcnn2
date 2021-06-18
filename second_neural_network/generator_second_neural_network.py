import numpy
import os
import torch as torch
import torch.nn as nn
import torch.nn.functional as F

from utils.node_object_creator import *
from layers.coding_layer import Coding_layer
from layers.convolutional_layer import Convolutional_layer
from layers.pooling_layer import Pooling_layer
from layers.dynamic_pooling import Max_pooling_layer, Dynamic_pooling_layer
from layers.hidden_layer import Hidden_layer
from second_neural_network.second_neural_network import SecondNeuralNetwork
from utils.utils import writer, plot_confusion_matrix, conf_matrix, accuracy, bad_predicted_files


class Generator_second_neural_network(SecondNeuralNetwork):

    def __init__(self, device, pattern, n = 20, m = 4, pooling = 'one-way pooling'):
        super().__init__(device, n, m)
        self.pooling = pooling
        self.pattern = pattern

    
    def matrices_and_layers_initialization(self):
        # Initialize the layers
        self.conv = Convolutional_layer(self.vector_size, self.device, features_size = self.feature_size)
        self.hidden = Hidden_layer(self.feature_size)
        if self.pooling == 'three-way pooling':
            self.dynamic = Dynamic_pooling_layer()
            self.max_pool = Max_pooling_layer()
        else:
            self.pooling = Pooling_layer()

        # Initialize matrices and bias
        self.w_t, self.w_l, self.w_r, self.b_conv = self.conv.initialize_matrices_and_bias()
        if self.pooling == 'three-way pooling':
            self.w_hidden, self.b_hidden = self.hidden.initialize_matrices_and_bias()
        else:
            self.w_hidden, self.b_hidden = self.hidden.initialize_matrices_and_bias()

        params = [self.w_t, self.w_l, self.w_r, self.b_conv, self.w_hidden, self.b_hidden]

        return params


    def layers(self, vector_representation_params):
        # Parameters of the first neural network
        ls_nodes, w_l_code, w_r_code, b_code = vector_representation_params
        del w_l_code
        del w_r_code
        del b_code
        ls_nodes = self.conv.convolutional_layer(ls_nodes)
        if self.pooling == 'three-way pooling':
            dict_sibling = None
            self.max_pool.max_pooling(ls_nodes)
            vector = self.dynamic.three_way_pooling(ls_nodes, dict_sibling)
        else:
            vector = self.pooling.pooling_layer(ls_nodes)
        del ls_nodes
        output = self.hidden.hidden_layer(vector)
        del vector

        return output


    def save(self):
        '''Save all the trained parameters into a csv file'''
        directory = os.path.join('params', self.pattern)
        if not os.path.exists(directory):
            os.mkdir(directory)

        # save w_t into csv file
        w_t = self.w_t.detach().numpy()
        numpy.savetxt(os.path.join(directory, "w_t.csv"), w_t, delimiter = ",")

        # save w_l into csv file
        w_l = self.w_l.detach().numpy()
        numpy.savetxt(os.path.join(directory, "w_l.csv"), w_l, delimiter = ",")
        
        # save w_r into csv file
        w_r = self.w_r.detach().numpy()
        numpy.savetxt(os.path.join(directory, "w_r.csv"), w_r, delimiter = ",")

        # save b_conv into csv file
        b_conv = self.b_conv.detach().numpy()
        numpy.savetxt(os.path.join(directory, "b_conv.csv"), b_conv, delimiter = ",")

        # save w_hidden into csv file
        w_hidden = self.w_hidden.detach().numpy()
        numpy.savetxt(os.path.join(directory, "w_hidden.csv"), w_hidden, delimiter = ",")

        # save b_conv into csv file
        b_hidden = self.b_hidden.detach().numpy()
        numpy.savetxt(os.path.join(directory, "b_hidden.csv"), b_hidden, delimiter = ",")
