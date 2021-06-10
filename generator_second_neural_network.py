from second_neural_network import SecondNeuralNetwork
import numpy
import os
import torch as torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import pickle
import gc

from node_object_creator import *
from coding_layer import Coding_layer
from convolutional_layer import Convolutional_layer
from pooling_layer import Pooling_layer
from dynamic_pooling import Max_pooling_layer, Dynamic_pooling_layer
from hidden_layer import Hidden_layer
from utils import writer, plot_confusion_matrix, conf_matrix, accuracy, bad_predicted_files


class Generator_second_neural_network(SecondNeuralNetwork):

    def __init__(self, device, n = 20, m = 4, pooling = 'one-way pooling'):
        super().__init__(device, n, m, pooling)

    def layers(self, vector_representation_params):
        ls_nodes, w_l_code, w_r_code, b_code = vector_representation_params
        del w_l_code
        del w_r_code
        del b_code
        ls_nodes = self.conv.convolutional_layer(ls_nodes, self.w_t, self.w_r, self.w_l, self.b_conv)
        if self.pooling == 'three-way pooling':
            dict_sibling = None
            self.max_pool.max_pooling(ls_nodes)
            vector = self.dynamic.three_way_pooling(ls_nodes, dict_sibling)
        else:
            vector = self.pooling.pooling_layer(ls_nodes)
        del ls_nodes
        output = self.hidden.hidden_layer(vector, self.w_hidden, self.b_hidden)
        del vector

        return output


    def save(self):
        '''Save all the trained parameters into a csv file'''
        directory = os.path.join('params', 'generators')
        if not os.path.exists(directory):
            os.mkdir(directory)

        # save w_comb1 into csv file
        w_comb1 = self.w_comb1.detach().numpy()
        numpy.savetxt(os.path.join(directory, "w_comb1.csv"), w_comb1, delimiter = ",")

        # save w_comb2 into csv file
        w_comb2 = self.w_comb2.detach().numpy()
        numpy.savetxt(os.path.join(directory, "w_comb2.csv"), w_comb2, delimiter = ",")

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
