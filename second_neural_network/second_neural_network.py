import numpy
import os
import torch as torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import pickle
import gc

from utils.node_object_creator import *
from first_neural_network.node import Node
from layers.coding_layer import Coding_layer
from layers.convolutional_layer import Convolutional_layer
from layers.pooling_layer import Pooling_layer
from layers.dynamic_pooling import Max_pooling_layer, Dynamic_pooling_layer
from layers.hidden_layer import Hidden_layer
from utils.utils import writer, plot_confusion_matrix, conf_matrix, accuracy, bad_predicted_files



class SecondNeuralNetwork(nn.Module):

    def __init__(self, device, n = 20, m = 4):
        ###############################
        super(SecondNeuralNetwork, self).__init__()
        ###############################
        self.vector_size = n
        self.feature_size = m
        #we create an attribute for the best accuracy so far (initialized to 0)
        self.best_accuracy = 0
        #device
        self.device = device


    def train(self, training_generator, validation_generator, total_epochs = 40, learning_rate = 0.01, batch_size = 20):
        """Create the training loop"""
        # Construct the optimizer
        #params = [self.w_comb1, self.w_comb2, self.w_t, self.w_l, self.w_r, self.b_conv, self.w_hidden, self.b_hidden]
        params = self.matrices_and_layers_initialization()
        optimizer = torch.optim.SGD(params, lr = learning_rate)
        criterion = nn.BCEWithLogitsLoss()
        print('Entering the neural network')
        #print('The correct value of the files is: ', targets)
        best_epoch = 0
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
                target.to(self.device)
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

            
            if accuracy > self.best_accuracy:
                    #we only save the paramters that provide the best accuracy
                    self.best_accuracy = accuracy
                    best_epoch = epoch
                    self.save()
            

        message = f'''
The accuracy we have is: {self.best_accuracy} for epoch {best_epoch}
        '''
        writer(message)


    def matrices_and_layers_initialization(self):
        pass
        

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
        validation_loss, accuracy_value, predicts, validation_targets, validation_set = self.forward_validation(validation_generator)

        print('Validation accuracy: ', accuracy_value)
        
        # Confusion matrix
        confusion_matrix = conf_matrix(predicts, validation_targets)
        print('Confusión matrix: ')
        print(confusion_matrix)

        files_bad_predicted = bad_predicted_files(validation_set, predicts, validation_targets)
        print(files_bad_predicted)

        if accuracy_value > self.best_accuracy:
            plot_confusion_matrix(confusion_matrix, ['no generator', 'generator'], lr2 = learning_rate, feature_size = self.feature_size, epoch = epoch)
    
        return validation_loss, accuracy_value


    def forward_validation(self, validation_generator):
        criterion = nn.BCEWithLogitsLoss()
        validation_set = []
        softmax = nn.Sigmoid()
        validation_loss = 0
        errors = 0
        number_of_files = 0
        all_predicts = torch.empty(0)
        all_targets = torch.empty(0)
        with torch.set_grad_enabled(False):
            for batch, target in validation_generator:
                target.to(self.device)
                predicts = torch.empty(0)
                outputs = torch.empty(0)
                for file in batch: 
                    validation_set.append(file)
                    with open(file, 'rb') as f:
                        data = pickle.load(f)
                    number_of_files += 1
                
                    ## forward (layers calculations)
                    output = self.layers(data)
                    del data
                    predicts = torch.cat((predicts, softmax(output)), 0)
                    outputs = torch.cat((outputs, output), 0)

                target = target.float()
                loss = criterion(outputs, target)
                accuracy_value = accuracy(predicts, target)
                #ahora 'accuracy value' no es la función accuracy, sino la cantidad absoluta de errores sobre el total del batch
                errors += accuracy_value
                validation_loss += loss.item()*len(batch)

                #añadimos los predicts y targets a los tensores que contienen toda la información
                all_predicts = torch.cat((all_predicts, predicts), 0)
                all_targets = torch.cat((all_targets, target), 0)

        gc.collect()
        total_accuracy = (number_of_files - float(errors))/number_of_files
        return validation_loss, total_accuracy, all_predicts, all_targets, validation_set


    def layers(self, vector_representation_params):
        pass

    def save(self):
        pass
