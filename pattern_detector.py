import os
import pandas as pd
import torch as torch
import torch.nn as nn
from time import time
import shutil
from git import rmtree
import click
import importlib

from utils.node_object_creator import *
from first_neural_network.first_neural_network import First_neural_network
from layers.coding_layer import Coding_layer
from layers.convolutional_layer import Convolutional_layer
from layers.dynamic_pooling import Dynamic_pooling_layer, Max_pooling_layer
from layers.pooling_layer import Pooling_layer
from layers.hidden_layer import Hidden_layer
from utils.repos import download_repos
from get_input import Get_input


class Pattern_detection():

    def __init__(self, pattern):
        self.pattern = pattern
        self.vector_size = self.set_vector_size()


    def pattern_detection(self, path):
        
        # Load the trained matrices and vectors
        self.load_matrices_and_vectors()

        # Data dictionary creation
        data_dict = self.data_dict_creation(path)
        #print('El primer data_dcit : ', data_dict)

        # Training the first neural network
        data_dict = self.first_neural_network(data_dict)

        # We calculate the predictions
        predicts = self.prediction(data_dict)
        
        # We print the predictions
        message = self.print_predictions(predicts, data_dict)
        print(message)
        return message


    def set_vector_size(self):
        df = pd.read_csv(os.path.join('initial_parameters', self.pattern, 'initial_vector_representation.csv'))
        vector_size = len(df[df.columns[0]])

        return vector_size

    
    def load_matrices_and_vectors(self):
        pass


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


    def first_neural_network(self, data_dict):
        # We have to import the parameters used to trained the first neural network
        module = importlib.import_module('initial_parameters.' + self.pattern + '.parameters_first_neural_network')
        vector_size = getattr(module, 'vector_size')
        learning_rate = getattr(module, 'learning_rate')
        momentum = getattr(module,'momentum')
        l2_penalty = getattr(module, 'l2_penalty')
        epoch = getattr(module, 'epoch_first')
        
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
            set_vector(ls_nodes, self.pattern)
            # Calculate the vector representation for each node
            vector_representation = First_neural_network(ls_nodes, vector_size, learning_rate, momentum, l2_penalty, epoch)

            # Calculate the vector representation for each node
            params = vector_representation.train()
            #we get the necessary input for the second neural network
            get_input_second_cnn = Get_input(ls_nodes, self.vector_size)
            get_input_second_cnn.get_input()
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
        pass


    def print_predictions(self, predicts, data_dict):
        i = 0
        message = ''
        for data in data_dict.keys():
            if predicts[i] < 0.5:
                path = data.split(os.path.sep)
                path.pop(0)
                name = os.path.join(*(path))
                message = message + '<p> The file ' + name + ' has not ' + self.pattern + 's</p>'
            else:
                path = data.split(os.path.sep)
                path.pop(0)
                name = os.path.join(*(path))
                message = message + '<p> The file ' + name + ' has ' + self.pattern + 's</p>'
            i+=1
        return message


def set_leaves(ls_nodes):
    for node in ls_nodes:
        node.set_leaves()


def set_vector(ls_nodes, pattern):
    df = pd.read_csv(os.path.join('initial_parameters', pattern, 'initial_vector_representation.csv'))
    for node in ls_nodes:
        node.set_vector(df)


@click.command()
@click.argument('pattern', required = True, nargs=1, type=str)
def main(pattern):
    welcome_message = '''
        ---------------------------------------------------------------------------------
    This is the Discern program. The main objective is to be able to find if the files within a project
    contain any generators. You can either input the path to a folder locally, or you can indicate 
    a URL from a github project
    -----------------------------------------------------------------------------
    '''
    print(welcome_message)
    get_input(pattern)


def get_input(pattern):
    if pattern_exists(pattern):
        #We instantiate the subclass for this pattern
        class_name = pattern.capitalize() + '_detection'
        module = importlib.import_module('pattern_detection.' + pattern + '_detector')
        pattern_class = getattr(module, class_name)
        pattern_detector = pattern_class(pattern)

        choice = '''

        Do you wish to indicate the path to a local folder ([y] / n) ?: 
        '''
        print(choice)
        x = input()
        if x:
            if x == 'y':
                print('Please indicate the path to the folder')
                x = input()
                pattern_detector.pattern_detection(x)
            elif x == 'n':
                choose_url(pattern)
            else:
                print('Invalid expression')
                get_input(pattern)
        else:
            print('Please indicate the path to the folder')
            x = input()
            pattern_detector.pattern_detection(x)

    else:
        message = '''
        ---------------------------------------------------------------------------------
        This pattern is not implemented. Please check the following:
            - There is a second neural network subclass implemented for this pattern.
            - The pattern name is well written.
        -----------------------------------------------------------------------------
        '''
        print(message)
        sys.exit()


def pattern_exists(pattern):
    file_name = os.path.join('pattern_detection', pattern + '_detector.py')
    if os.path.isfile(file_name):
        return True
    else:
        return False


def choose_url(pattern):
    choice = '''
    Then do you wish to indicate a URL? ([y] / n)?:
    '''
    print(choice)
    x = input()
    if x:
        if x == 'y':
            print('Please indicate the URL: ')
            x = input()
            validate_from_url(x, pattern)
        elif x == 'n':
            print('Exiting program')
        else:
            print('Invalid expression')
            choose_url(pattern)
    else:
        print('Please indicate the URL: ')
        x = input()
        validate_from_url(x, pattern)


def validate_from_url(url, pattern):
    download_repos([url], 'downloaded_validate')
    path = get_path(url)
    #We instantiate the subclass for this pattern
    class_name = pattern.capitalize() + '_detection'
    module = importlib.import_module('pattern_detection.' + pattern + '_detector')
    pattern_class = getattr(module, class_name)
    pattern_detector = pattern_class(pattern)
    pattern_detector.pattern_detection(path)
    folder_deleter(path)


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