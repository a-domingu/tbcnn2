import os
from time import time
import pandas as pd
import pickle
import gc

from node_object_creator import *
from first_neural_network import First_neural_network


    

def main(path, vector_size , learning_rate, momentum, l2_penalty, epoch):
    # Training the first neural network
    i = 1
    for tree in read_folder_data_set(path):
        time1 = time()

        # convert its nodes into the Node class we have, and assign their attributes
        main_node = node_object_creator(tree)
        ls_nodes = main_node.descendants()
        del main_node

        # We assign the leaves nodes under each node
        set_leaves(ls_nodes)
        # Initializing vector embeddings
        set_vector(ls_nodes)
        '''
        print('Initial vectors: ')

        for node in ls_nodes:
            print(node.vector)
            break

        for node in main_node.descendants():
            print(node.vector)
            print('####################')
            break
        '''
        # Calculate the vector representation for each node
        vector_representation = First_neural_network(ls_nodes, vector_size, learning_rate, momentum, l2_penalty, epoch)
        ls_nodes, w_l_code, w_r_code, b_code = vector_representation.vector_representation()
        '''
        print('After first neural network: ')

        for node in ls_nodes:
            print(node.vector)
            break

        for node in main_node.descendants():
            print(node.vector)
            print('####################')
            break
        '''
        filename = os.path.join('vector_representation', os.path.basename(tree) + '.txt')
        params = [ls_nodes, w_l_code, w_r_code, b_code]
        del ls_nodes
        del  w_l_code
        del w_r_code
        del b_code

        with open(filename, 'wb') as f:
            pickle.dump(params, f)
        
        del params

        time2= time()
        dtime = time2 - time1
        print(f"Vector rep. of file: {tree} {i} in ", dtime//60, 'min and', dtime%60, 'sec.')
        '''
        with open(filename, 'rb') as f:
            params_2 = pickle.load(f) 

        ls_nodes_2 = params_2[0]

        print('After save vectors with pickle: ')
        
        for node in ls_nodes_2:
            print(node.vector)
            break

        main_node_2 = ls_nodes_2[0]
        for node in main_node_2.descendants():
            print(node.vector)
            print('#################### \n')
            break
        '''
        if (i%50 == 0):
            gc.collect()
        i += 1


def read_folder_data_set(path):
    # iterates through the generators directory, identifies the folders and enter in them
    for (dirpath, _dirnames, filenames) in os.walk(path):
        if dirpath.endswith('withgen') or dirpath.endswith('nogen'):
            for filename in filenames:
                if filename.endswith('.py'):
                    filepath = os.path.join(dirpath, filename)
                    yield filepath


def set_leaves(ls_nodes):
    for node in ls_nodes:
        node.set_leaves()

def set_vector(ls_nodes):
    df = pd.read_csv('initial_vector_representation.csv')
    for node in ls_nodes:
        node.set_vector(df)

########################################

if __name__ == '__main__':
    # Folder path
    path = os.path.join('sets_short', 'generators')
    # First neural network parameters
    vector_size = 30
    learning_rate = 0.3
    momentum = 0
    l2_penalty = 0
    epoch = 1

    main(path, vector_size, learning_rate, momentum, l2_penalty, epoch)