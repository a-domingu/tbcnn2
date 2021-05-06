import sys
import os
import gensim
import random
import numpy
import pandas as pd
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
from utils import writer


class SecondNeuralNetwork():

    def __init__(self, n = 20, m = 4, pooling = 'one-way pooling'):
        self.vector_size = n
        self.feature_size = m
        # parameters
        self.w_comb1 = torch.diag(torch.randn(self.vector_size, dtype=torch.float32)).requires_grad_()
        self.w_comb2 = torch.diag(torch.randn(self.vector_size, dtype=torch.float32)).requires_grad_()
        self.w_t = torch.randn(self.feature_size, self.vector_size, requires_grad = True)
        self.w_r = torch.randn(self.feature_size, self.vector_size, requires_grad = True)
        self.w_l = torch.randn(self.feature_size, self.vector_size, requires_grad = True)
        self.b_conv = torch.randn(self.feature_size, requires_grad = True)
        # pooling method
        self.pooling = pooling
        if self.pooling == 'three-way pooling':
            self.w_hidden = torch.randn(3, requires_grad = True)
            self.b_hidden = torch.randn(1, requires_grad = True)
            self.dynamic = Dynamic_pooling_layer()
            self.max_pool = Max_pooling_layer()
        else:
            self.w_hidden = torch.randn(self.feature_size, requires_grad = True)
            self.b_hidden = torch.randn(1, requires_grad = True)
            self.pooling = Pooling_layer()
        # layers
        self.cod = Coding_layer(self.vector_size, self.w_comb1, self.w_comb2)
        self.conv = Convolutional_layer(self.vector_size, self.w_t, self.w_r, self.w_l, self.b_conv, features_size=self.feature_size)
        self.hidden = Hidden_layer(self.w_hidden, self.b_hidden)


    

    def train(self, targets, training_dict, total_epochs = 10, learning_rate = 0.1):
        """Create the training loop"""
        # Construct the optimizer
        params = [self.w_comb1, self.w_comb2, self.w_t, self.w_l, self.w_r, self.b_conv, self.w_hidden, self.b_hidden]
        optimizer = torch.optim.SGD(params, lr = learning_rate)
        criterion = nn.BCELoss()
        print('The correct value of the files is: ', targets)

        for epoch in range(total_epochs):
            # Time
            start = time()
            outputs = self.forward(training_dict)

            try:
                loss = criterion(outputs, targets)
            except AttributeError:
                print(f'The size of outputs is {len(outputs)} and is of type {type(outputs)}')
                print('Check that the path is a folder and not a file')
                raise AttributeError
            
            # zero the parameter gradients
            print('outputs: \n', outputs)
            #print('Matrix w_r_conv: \n', params[4])

            optimizer.zero_grad()

            # Calculates the derivative
            loss.backward(retain_graph = True)

            # Update parameters
            optimizer.step() #w_r = w_r - lr * w_r.grad

            #Time
            end = time()

            print('Epoch: ', epoch, ', Time: ', end-start, ', Loss: ', loss)

        message = f'''
The loss we have for the training network is: {loss}
        '''
        writer(message)
        self.save()


    def forward(self, training_dict):
        outputs = []
        softmax = nn.Sigmoid()
        for filepath in training_dict.keys():
            data = filepath
            
            ## forward (layers calculations)
            output = self.layers(training_dict[data])

            # output append
            if outputs == []:
                outputs = softmax(output)
            else:
                outputs = torch.cat((outputs, softmax(output)), 0)

        return outputs


    def layers(self, vector_representation_params):
        ls_nodes = vector_representation_params[0]
        dict_ast_to_Node = vector_representation_params[1]
        dict_sibling = vector_representation_params[2]
        w_l_code = vector_representation_params[3]
        w_r_code = vector_representation_params[4]
        b_code = vector_representation_params[5]
        ls_nodes = self.cod.coding_layer(ls_nodes, dict_ast_to_Node, w_l_code, w_r_code, b_code)
        ls_nodes = self.conv.convolutional_layer(ls_nodes, dict_ast_to_Node)
        if self.pooling == 'three-way pooling':
            self.max_pool.max_pooling(ls_nodes)
            vector = self.dynamic.three_way_pooling(ls_nodes, dict_sibling)
        else:
            vector = self.pooling.pooling_layer(ls_nodes)
        output = self.hidden.hidden_layer(vector)

        return output


    def save(self):
        '''Save all the trained parameters into a csv file'''
        #parameters = pd.DataFrame({'w_comb1': self.w_comb1.detach().numpy(), 'w_comb2': self.w_comb2.detach().numpy(), 'w_t': self.w_t.detach().numpy(), 'w_l': self.w_l.detach().numpy(), 'w_r': self.w_r.detach().numpy(), 'b_conv': self.b_conv.detach().numpy(), 'w_hidden': self.w_hidden.detach().numpy(), 'b_hidden': self.b_hidden.detach().numpy()})
        # save w_comb1 into csv file
        w_comb1 = self.w_comb1.detach().numpy()
        numpy.savetxt("params\\w_comb1.csv", w_comb1, delimiter = ",")

        # save w_comb2 into csv file
        w_comb2 = self.w_comb2.detach().numpy()
        numpy.savetxt("params\\w_comb2.csv", w_comb2, delimiter = ",")

        # save w_t into csv file
        w_t = self.w_t.detach().numpy()
        numpy.savetxt("params\\w_t.csv", w_t, delimiter = ",")

        # save w_l into csv file
        w_l = self.w_l.detach().numpy()
        numpy.savetxt("params\\w_l.csv", w_l, delimiter = ",")
        
        # save w_r into csv file
        w_r = self.w_r.detach().numpy()
        numpy.savetxt("params\\w_r.csv", w_r, delimiter = ",")

        # save b_conv into csv file
        b_conv = self.b_conv.detach().numpy()
        numpy.savetxt("params\\b_conv.csv", b_conv, delimiter = ",")

        # save w_hidden into csv file
        w_hidden = self.w_hidden.detach().numpy()
        numpy.savetxt("params\\w_hidden.csv", w_hidden, delimiter = ",")

        # save b_conv into csv file
        b_hidden = self.b_hidden.detach().numpy()
        numpy.savetxt("params\\b_hidden.csv", b_hidden, delimiter = ",")