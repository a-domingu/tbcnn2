import os
from pattern_test import Pattern_test
import numpy
import pandas as pd
import torch as torch


from node_object_creator import *
from first_neural_network import First_neural_network
from coding_layer import Coding_layer
from convolutional_layer import Convolutional_layer
from dynamic_pooling import Dynamic_pooling_layer, Max_pooling_layer
from pooling_layer import Pooling_layer
from hidden_layer import Hidden_layer
from utils import conf_matrix, accuracy, bad_predicted_files


class Generator_test(Pattern_test):

    def __init__(self, pooling = 'one-way pooling'):
        super().__init__()
        self.feature_size = self.set_feature_size()
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


    def set_feature_size(self):
        df = pd.read_csv(os.path.join('params', 'generator', 'w_t.csv'))
        feature_size = len(df[df.columns[0]])

        return feature_size

    
    def load_matrices_and_vectors(self):
        '''Load all the trained parameters from a csv file'''
        directory = os.path.join('params', 'generator')
        if not os.path.exists(directory):
            os.mkdir(directory)

        w_comb1 = numpy.genfromtxt(os.path.join(directory, "w_comb1.csv"), delimiter = ",")
        self.w_comb1 = torch.tensor(w_comb1, dtype=torch.float32)

        w_comb2 = numpy.genfromtxt(os.path.join(directory, "w_comb2.csv"), delimiter = ",")
        self.w_comb2 = torch.tensor(w_comb2, dtype=torch.float32)

        w_t = numpy.genfromtxt(os.path.join(directory, "w_t.csv"), delimiter = ",")
        self.w_t = torch.tensor(w_t, dtype=torch.float32)

        w_r = numpy.genfromtxt(os.path.join(directory, "w_r.csv"), delimiter = ",")
        self.w_r = torch.tensor(w_r, dtype=torch.float32)

        w_l = numpy.genfromtxt(os.path.join(directory, "w_l.csv"), delimiter = ",")
        self.w_l = torch.tensor(w_l, dtype=torch.float32)

        b_conv = numpy.genfromtxt(os.path.join(directory, "b_conv.csv"), delimiter = ",")
        self.b_conv = torch.tensor(b_conv, dtype=torch.float32)

        w_hidden = numpy.genfromtxt(os.path.join(directory, "w_hidden.csv"), delimiter = ",")
        self.w_hidden = torch.tensor(w_hidden, dtype=torch.float32)

        b_hidden = numpy.genfromtxt(os.path.join(directory, "b_hidden.csv"), delimiter = ",")
        self.b_hidden = torch.tensor(b_hidden, dtype=torch.float32)


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


def set_leaves(ls_nodes):
    for node in ls_nodes:
        node.set_leaves()

def set_vector(ls_nodes):
    df = pd.read_csv('initial_vector_representation.csv')
    for node in ls_nodes:
        node.set_vector(df)
        

