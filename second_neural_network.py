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
        # Create uniform random numbers in half-open interval [-1.0, 1.0)
        r1 = -1.0
        r2 = 1.0
        self.w_comb1 = torch.diag(torch.squeeze(torch.distributions.Uniform(-1, +1).sample((self.vector_size, 1)), 1)).requires_grad_()
        #self.w_comb1 = torch.diag(torch.rand(self.vector_size, dtype=torch.float32)).requires_grad_()
        self.w_comb2 = torch.diag(torch.squeeze(torch.distributions.Uniform(-1, +1).sample((self.vector_size, 1)), 1)).requires_grad_()
        #self.w_comb2 = torch.diag(torch.rand(self.vector_size, dtype=torch.float32)).requires_grad_()
        self.w_t = torch.distributions.Uniform(-1, +1).sample((self.feature_size, self.vector_size)).requires_grad_()
        #self.w_t = torch.rand(self.feature_size, self.vector_size, requires_grad = True)
        self.w_r = torch.distributions.Uniform(-1, +1).sample((self.feature_size, self.vector_size)).requires_grad_()
        #self.w_r = torch.rand(self.feature_size, self.vector_size, requires_grad = True)
        self.w_l = torch.distributions.Uniform(-1, +1).sample((self.feature_size, self.vector_size)).requires_grad_()
        #self.w_l = torch.rand(self.feature_size, self.vector_size, requires_grad = True)
        self.b_conv = torch.squeeze(torch.distributions.Uniform(-1, +1).sample((self.feature_size, 1))).requires_grad_()
        #self.b_conv = torch.rand(self.feature_size, requires_grad = True)
        # pooling method
        self.pooling = pooling
        if self.pooling == 'three-way pooling':
            self.w_hidden = torch.rand(3, requires_grad = True)
            self.b_hidden = torch.rand(1, requires_grad = True)
            self.dynamic = Dynamic_pooling_layer()
            self.max_pool = Max_pooling_layer()
        else:
            self.w_hidden = torch.squeeze(torch.distributions.Uniform(-1, +1).sample((self.feature_size, 1))).requires_grad_()
            #self.w_hidden = torch.rand(self.feature_size, requires_grad = True)
            self.b_hidden = torch.rand(1, requires_grad = True)
            self.pooling = Pooling_layer()
        # layers
        self.cod = Coding_layer(self.vector_size)
        self.conv = Convolutional_layer(self.vector_size, features_size=self.feature_size)
        self.hidden = Hidden_layer()


    

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

            # zero the parameter gradients
            optimizer.zero_grad()
            '''
            w_comb1_big = torch.gt(self.w_comb1, 1.0).sum()
            w_comb1_small = torch.gt(torch.neg(self.w_comb1), 1.0).sum()
            print('###############')
            print('Matrix w_comb1:')
            print('Number of elements bigger than 1: ', w_comb1_big)
            print('Number of elements smaller than -1: ', w_comb1_small)
            w_comb2_big = torch.gt(self.w_comb2, 1.0).sum()
            w_comb2_small = torch.gt(torch.neg(self.w_comb2), 1.0).sum()
            print('###############')
            print('Matrix w_comb2:')
            print('Number of elements bigger than 1: ', w_comb2_big)
            print('Number of elements smaller than -1: ', w_comb2_small)
            print('###############')
            w_t_big = torch.gt(self.w_t, 1.0).sum()
            w_t_small = torch.gt(torch.neg(self.w_t), 1.0).sum()
            print('Matrix w_t:')
            print('Number of elements bigger than 1: ', w_t_big)
            print('Number of elements smaller than -1: ', w_t_small)
            print('###############')
            w_r_big = torch.gt(self.w_r, 1.0).sum()
            w_r_small = torch.gt(torch.neg(self.w_r), 1.0).sum()
            print('Matrix w_r:')
            print('Number of elements bigger than 1: ', w_r_big)
            print('Number of elements smaller than -1: ', w_r_small)
            print('###############')
            w_l_big = torch.gt(self.w_l, 1.0).sum()
            w_l_small = torch.gt(torch.neg(self.w_l), 1.0).sum()
            print('Matrix w_l:')
            print('Number of elements bigger than 1: ', w_l_big)
            print('Number of elements smaller than -1: ', w_l_small)
            print('###############')
            b_conv_big = torch.gt(self.b_conv, 1.0).sum()
            b_conv_small = torch.gt(torch.neg(self.b_conv), 1.0).sum()
            print('Matrix b_conv:')
            print('Number of elements bigger than 1: ', b_conv_big)
            print('Number of elements smaller than -1: ', b_conv_small)
            print('###############')
            w_hidden_big = torch.gt(self.w_hidden, 1.0).sum()
            w_hidden_small = torch.gt(torch.neg(self.w_hidden), 1.0).sum()
            print('Matrix w_hidden:')
            print('Number of elements bigger than 1: ', w_hidden_big)
            print('Number of elements smaller than -1: ', w_hidden_small)
            print('###############')
            b_hidden_big = torch.gt(self.b_hidden, 1.0).sum()
            b_hidden_small = torch.gt(torch.neg(self.b_hidden), 1.0).sum()
            print('Vector b_hidden:')
            print('Number of elements bigger than 1: ', b_hidden_big)
            print('Number of elements smaller than -1: ', b_hidden_small)
            print('###############')
            '''
            # forward
            outputs = self.forward(training_dict)
            print('Outputs: \n', outputs)

            # Computes the loss function
            try:
                loss = criterion(outputs, targets)
            except AttributeError:
                print(f'The size of outputs is {len(outputs)} and is of type {type(outputs)}')
                print('Check that the path is a folder and not a file')
                raise AttributeError

            # Backward = calculates the derivative
            loss.backward() # w_r.grad = dloss/dw_r
            for i in range(8):
                print(params[i].grad.sum())

            # Update parameters
            optimizer.step() #w_r = w_r - lr * w_r.grad

            #Time
            end = time()

            print('Epoch: ', epoch, ', Time: ', end-start, ', Loss: ', loss)
            print('############### \n')

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
        ls_nodes = self.cod.coding_layer(ls_nodes, dict_ast_to_Node, w_l_code, w_r_code, b_code, self.w_comb1, self.w_comb2)
        ls_nodes = self.conv.convolutional_layer(ls_nodes, dict_ast_to_Node, self.w_t, self.w_r, self.w_l, self.b_conv)
        if self.pooling == 'three-way pooling':
            self.max_pool.max_pooling(ls_nodes)
            vector = self.dynamic.three_way_pooling(ls_nodes, dict_sibling)
        else:
            vector = self.pooling.pooling_layer(ls_nodes)
        '''
        vector_big = torch.gt(vector, 4.0).sum()
        vector_small = torch.gt(torch.neg(vector), 4.0).sum()
        print('vector pooling:')
        print('Number of elements bigger than 4: ', vector_big)
        print('Number of elements smaller than -4: ', vector_small)
        print('###############')
        '''
        output = self.hidden.hidden_layer(vector, self.w_hidden, self.b_hidden)

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