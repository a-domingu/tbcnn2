import os
import numpy
import pandas as pd
import torch as torch
import torch.nn as nn
from time import time
import shutil
from git import rmtree

from node_object_creator import *
from first_neural_network import First_neural_network
from coding_layer import Coding_layer
from convolutional_layer import Convolutional_layer
from dynamic_pooling import Dynamic_pooling_layer, Max_pooling_layer
from pooling_layer import Pooling_layer
from hidden_layer import Hidden_layer
from repos import download_repos


class Generator_pattern_detection():

    def __init__(self, n = 30, m = 100, pooling = 'one-way pooling'):
        self.vector_size = n
        self.feature_size = m
        # parameters
        w_comb1 = numpy.genfromtxt(os.path.join("params","w_comb1.csv"), delimiter = ",")
        self.w_comb1 = torch.tensor(w_comb1, dtype=torch.float32)
        w_comb2 = numpy.genfromtxt(os.path.join("params","w_comb2.csv"), delimiter = ",")
        self.w_comb2 = torch.tensor(w_comb2, dtype=torch.float32)
        w_t = numpy.genfromtxt(os.path.join("params","w_t.csv"), delimiter = ",")
        self.w_t = torch.tensor(w_t, dtype=torch.float32)
        w_r = numpy.genfromtxt(os.path.join("params","w_r.csv"), delimiter = ",")
        self.w_r = torch.tensor(w_r, dtype=torch.float32)
        w_l = numpy.genfromtxt(os.path.join("params","w_l.csv"), delimiter = ",")
        self.w_l = torch.tensor(w_l, dtype=torch.float32)
        b_conv = numpy.genfromtxt(os.path.join("params","b_conv.csv"), delimiter = ",")
        self.b_conv = torch.tensor(b_conv, dtype=torch.float32)
        w_hidden = numpy.genfromtxt(os.path.join("params","w_hidden.csv"), delimiter = ",")
        self.w_hidden = torch.tensor(w_hidden, dtype=torch.float32)
        b_hidden = numpy.genfromtxt(os.path.join("params","b_hidden.csv"), delimiter = ",")
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
        #message = '########################################<br>'
        #message = message + 'Doing the embedding for each file <br>'

        # Data dictionary creation
        data_dict = self.data_dict_creation(path)
        #print('El primer data_dcit : ', data_dict)

        # Training the first neural network
        data_dict = self.first_neural_network(data_dict)

        # We calculate the predictions
        predicts = self.prediction(data_dict)
        
        # We print the predictions
        message = self.print_predictions(predicts, data_dict)
        return message


    def data_dict_creation(self, path):
        '''we want to read a path that can be a file or a folder with .py files'''
        # we create the data dict with all the information about vector representation
        data_dict = {}
        if path.endswith('.py'):
            data_dict[path] = None
        else:
            # iterates through the generators directory, identifies the folders and enter in them
            for (dirpath, dirnames, filenames) in os.walk(path):
                for filename in filenames:
                    if filename.endswith('.py'):
                        filepath = os.path.join(dirpath, filename)
                        data_dict[filepath] = None
        return data_dict

    

    def first_neural_network(self, data_dict, vector_size = 30, learning_rate = 0.3, momentum = 0, l2_penalty = 0, epoch = 1):
        total = len(data_dict)
        i = 1
        for data in data_dict:
            time1 = time()

            # convert its nodes into the Node class we have, and assign their attributes
            main_node = node_object_creator(data)
            # we set the descendants of the main node and put them in a list
            ls_nodes = main_node.descendants()

            # We assign the leaves nodes under each node
            set_leaves(ls_nodes)
            # Initializing vector embeddings
            set_vector(ls_nodes)
            # Calculate the vector representation for each node
            vector_representation = First_neural_network(ls_nodes, vector_size, learning_rate, momentum, l2_penalty, epoch)

            # Calculate the vector representation for each node
            params = vector_representation.vector_representation()
            #params = [w_l_code, w_r_code, b_code]
            data_dict[data] = params
            time2= time()
            dtime = time2 - time1

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
        ls_nodes, w_l_code, w_r_code, b_code = vector_representation_params
        # we don't do coding layer and go directly to convolutional layer; that's why we don't
        # use the matrices above
        ls_nodes = self.conv.convolutional_layer(ls_nodes, self.w_t, self.w_r, self.w_l, self.b_conv)
        if self.pooling == 'one-way pooling':
            vector = self.pooling_layer.pooling_layer(ls_nodes)
        else:
            self.max_pool.max_pooling(ls_nodes)
            dict_sibling = {}
            vector = self.dynamic.three_way_pooling(ls_nodes, dict_sibling)
        output = self.hidden.hidden_layer(vector, self.w_hidden, self.b_hidden)

        return output


    def print_predictions(self, predicts, data_dict):
        i = 0
        message = ''
        for data in data_dict.keys():
            if predicts[i] < 0.5:
                path = data.split(os.path.sep)
                path.pop(0)
                name = os.path.join(*(path))
                message = message + '<p> The file ' + name + ' does not have generators</p>'
            else:
                path = data.split(os.path.sep)
                path.pop(0)
                name = os.path.join(*(path))
                message = message + '<p> The file ' + name + ' has generators</p>'
            i+=1
        return message


def set_leaves(ls_nodes):
    for node in ls_nodes:
        node.set_leaves()

def set_vector(ls_nodes):
    df = pd.read_csv('initial_vector_representation.csv')
    for node in ls_nodes:
        node.set_vector(df)


def main():
    welcome_message = '''
        ---------------------------------------------------------------------------------
    This is the Discern program. The main objective is to be able to find if the files within a project
    contain any generators. You can either input the path to a folder locally, or you can indicate 
    a URL from a github project
    -----------------------------------------------------------------------------
    '''
    print(welcome_message)
    get_input()



def get_input():
    generator_detector = Generator_pattern_detection()
    choice = '''

    Do you wish to indicate the path to a local folder ([y] / n) ?: 
    '''
    print(choice)
    x = input()
    if x:
        if x == 'y':
            print('Please indicate the path to the folder')
            x = input()
            generator_detector.generator_detection(x)
        elif x == 'n':
            choose_url()
        else:
            print('Invalid expression')
            get_input()
    else:
        print('Please indicate the path to the folder')
        x = input()
        generator_detector.generator_detection(x)


def choose_url():
    choice = '''
    Then do you wish to indicate a URL? ([y] / n)?:
    '''
    print(choice)
    x = input()
    if x:
        if x == 'y':
            print('Please indicate the URL: ')
            x = input()
            validate_from_url(x)
        elif x == 'n':
            print('Exiting program')
        else:
            print('Invalid expression')
            choose_url()
    else:
        print('Please indicate the URL: ')
        x = input()
        validate_from_url(x)


def validate_from_url(url):
    download_repos([url], 'downloaded_validate')
    path = get_path(url)
    generator_detector = Generator_pattern_detection()
    message = generator_detector.generator_detection(path)
    folder_deleter(path)
    return message


def get_path(url):
    project_name = url.split('/')[-1]
    path = os.path.join('downloaded_validate', project_name)
    return path

def folder_deleter(path):
    try:
        shutil.rmtree(path, ignore_errors=True)
        #os.remove(os.path.join(path, '.git'))
        rmtree(os.path.join(path, '.git'))
        os.rmdir(path)
    except Exception:
        print(f'Couldn\'t delete {path} folder')


if __name__ == '__main__':
    main()