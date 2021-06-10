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



class SecondNeuralNetwork(nn.Module):

    def __init__(self, device, n = 20, m = 4, pooling = 'one-way pooling'):
        ###############################
        super(SecondNeuralNetwork, self).__init__()
        ###############################
        self.vector_size = n
        self.feature_size = m
        # parameters
        # Create uniform random numbers in half-open interval [-1.0, 1.0)
        self.w_comb1 = torch.diag(torch.squeeze(torch.distributions.Uniform(-1, +1).sample((self.vector_size, 1)), 1)).requires_grad_()
        self.w_comb2 = torch.diag(torch.squeeze(torch.distributions.Uniform(-1, +1).sample((self.vector_size, 1)), 1)).requires_grad_()
        self.w_t = torch.distributions.Uniform(-1, +1).sample((self.feature_size, self.vector_size)).requires_grad_()
        self.w_r = torch.distributions.Uniform(-1, +1).sample((self.feature_size, self.vector_size)).requires_grad_()
        self.w_l = torch.distributions.Uniform(-1, +1).sample((self.feature_size, self.vector_size)).requires_grad_()
        self.b_conv = torch.squeeze(torch.distributions.Uniform(-1, +1).sample((self.feature_size, 1))).requires_grad_()
        # pooling method
        self.pooling = pooling
        if self.pooling == 'three-way pooling':
            self.w_hidden = torch.squeeze(torch.distributions.Uniform(-1, +1).sample((3, 1))).requires_grad_()
            self.b_hidden = torch.squeeze(torch.distributions.Uniform(-1, +1).sample()).requires_grad_()
            self.dynamic = Dynamic_pooling_layer()
            self.max_pool = Max_pooling_layer()
        else:
            self.w_hidden = torch.squeeze(torch.distributions.Uniform(-1, +1).sample((self.feature_size, 1))).requires_grad_()
            self.b_hidden = torch.rand(1, requires_grad = True)
            self.pooling = Pooling_layer()
        # layers
        self.cod = Coding_layer(self.vector_size)
        self.conv = Convolutional_layer(self.vector_size, features_size=self.feature_size)
        self.hidden = Hidden_layer()
        #we create an attribute for the best accuracy so far (initialized to 0)
        self.best_accuracy = 0
        #device
        self.device = device


    def train(self, training_generator, validation_generator, total_epochs = 40, learning_rate = 0.01, batch_size = 20):
        """Create the training loop"""
        # Construct the optimizer
        params = [self.w_comb1, self.w_comb2, self.w_t, self.w_l, self.w_r, self.b_conv, self.w_hidden, self.b_hidden]
        optimizer = torch.optim.SGD(params, lr = learning_rate)
        criterion = nn.BCEWithLogitsLoss()
        print('Entering the neural network')
        #print('The correct value of the files is: ', targets)

        for epoch in range(total_epochs):
            # Time
            start = time()

            sum_loss = 0
            nb_batch = 0
            train_loss = 0.0
            for data in training_generator:
                # Transfer to GPU
                #data = data.to(self.device)
                batch, target = data
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = self.forward(batch)

                # Computes the loss function
                #outputs = outputs.float()
                target = target.float()
                try:
                    loss = criterion(outputs, target)
                except AttributeError:
                    print(f'The size of outputs is {len(outputs)} and is of type {type(outputs)}')
                    print('Check that the path is a folder and not a file')
                    raise AttributeError

                # Backward = calculates the derivative
                loss.backward() # w_r.grad = dloss/dw_r
                sum_loss += loss.detach()

                # Update parameters
                optimizer.step() #w_r = w_r - lr * w_r.grad

                train_loss += loss.item()*len(batch)
                del loss

                nb_batch += 1

            #Time
            end = time()

            # Validation
            loss_validation, accuracy = self.validation(validation_generator, learning_rate, epoch)

            print('Epoch: ', epoch, ', Time: ', end-start, ', Training Loss: ', train_loss/len(training_generator), ', Validation Loss: ', loss_validation/len(validation_generator), ', accuracy: ', accuracy)
            print('############### \n')

            '''
            if accuracy_validation > self.best_accuracy:
                    #we only save the paramters that provide the best accuracy
                    self.best_accuracy = accuracy_validation
                    self.save()
            '''

        message = f'''
The loss we have for the training network is: {sum_loss/nb_batch}
        '''
        writer(message)
        

    def forward(self, batch_set):
        outputs = []
        #softmax = nn.Sigmoid()
        for data in batch_set:
            #filename = os.path.join('vector_representation', os.path.basename(data) + '.txt')
            with open(data, 'rb') as f:
                print('data: ', data)
                params_first_neural_network = pickle.load(f)
            
            ## forward (layers calculations)
            output = self.layers(params_first_neural_network)
            del params_first_neural_network

            # output append
            if outputs == []:
                #outputs = softmax(output)
                outputs = output
            else:
                #outputs = torch.cat((outputs, softmax(output)), 0)
                outputs = torch.cat((outputs, output), 0)

            del output

        gc.collect()
        return outputs


    def validation(self, validation_generator, learning_rate, epoch):
        # Test the accuracy of the updates parameters by using a validation set
        validation_loss, accuracy_value, predicts, validation_targets = self.forward_validation(validation_generator)

        print('Validation accuracy: ', accuracy_value)
        
        # Confusion matrix
        confusion_matrix = conf_matrix(predicts, validation_targets)
        print('Confusión matrix: ')
        print(confusion_matrix)
        #TODO preguntar cómo saber cuál es el conjunto de todos los archivos del validation set
        '''
        files_bad_predicted = bad_predicted_files(validation_dict, predicts, validation_targets)
        print(files_bad_predicted)
        '''
        if accuracy_value > self.best_accuracy:
            plot_confusion_matrix(confusion_matrix, ['no generator', 'generator'], lr2 = learning_rate, feature_size = self.feature_size, epoch = epoch)
    
        return validation_loss, accuracy_value


    def forward_validation(self, validation_generator):
        criterion = nn.BCELoss()
        outputs = []
        softmax = nn.Sigmoid()
        validation_loss = 0
        errors = 0
        number_of_files = 0
        predicts = []
        all_predicts = torch.empty(0)
        all_targets = torch.empty(0)
        with torch.set_grad_enabled(False):
            for batch, target in validation_generator:
                #data, target = data.to(self.device), target.to(self.device)
                predicts = []
                for file in batch: 
                    with open(file, 'rb') as f:
                        data = pickle.load(f)
                    number_of_files += 1
                
                ## forward (layers calculations)
                    output = self.layers(data)
                    del data
                    if predicts == []:
                        predicts = softmax(output)
                    else:
                        predicts = torch.cat((predicts, softmax(output)), 0)

                target = target.float()
                loss = criterion(predicts, target)
                accuracy_value = accuracy(predicts, target)
                #ahora 'accuracy value' no es la función accuracy, sino la cantidad absoluta de errores sobre el total del batch
                errors += accuracy_value
                validation_loss += loss.item()*len(batch)

                #añadimos los predicts y targets a los tensores que contienen toda la información
                all_predicts = torch.cat((all_predicts, predicts), 0)
                all_targets = torch.cat((all_targets, target), 0)

        gc.collect()
        total_accuracy = float(errors)/number_of_files
        return validation_loss, total_accuracy, all_predicts, all_targets


    def layers(self, vector_representation_params):
        pass

    def save(self):
        pass
