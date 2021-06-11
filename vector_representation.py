import os
import sys
from time import time
import pandas as pd
import pickle
import gc

from utils.node_object_creator import *
from first_neural_network.first_neural_network import First_neural_network


class Vector_representation():

    def __init__(self, folder, pattern, vector_size, learning_rate, momentum, l2_penalty, epoch):
        self.folder = folder
        self.pattern = pattern
        self.vector_size = vector_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.l2_penalty = l2_penalty
        self.epoch = epoch


    def vector_representation(self):
        # Training the first neural network
        i = 1
        for tree in self.read_folder_data_set():
            time1 = time()

            # convert its nodes into the Node class we have, and assign their attributes
            main_node = node_object_creator(tree)
            ls_nodes = main_node.descendants()
            del main_node

            # We assign the leaves nodes under each node
            self.set_leaves(ls_nodes)
            # Initializing vector embeddings
            self.set_vector(ls_nodes)

            # Calculate the vector representation for each node
            vector_representation = First_neural_network(ls_nodes, self.vector_size, self.learning_rate, self.momentum, self.l2_penalty, self.epoch)
            ls_nodes, w_l_code, w_r_code, b_code = vector_representation.train()

            filename = os.path.join('vector_representation', os.path.basename(tree) + '.txt')
            params = [ls_nodes, w_l_code, w_r_code, b_code]

            with open(filename, 'wb') as f:
                pickle.dump(params, f)
            
            del params
            del ls_nodes
            del  w_l_code
            del w_r_code
            del b_code

            time2= time()
            dtime = time2 - time1
            print(f"Vector rep. of file: {tree} {i} in ", dtime//60, 'min and', dtime%60, 'sec.')

            if (i%50 == 0):
                gc.collect()
            i += 1


    def read_folder_data_set(self):
        path = os.path.join(self.folder, self.pattern)
        #If there is not a set with the required pattern, we print an error
        if not os.path.isdir(path):
            message = '''
            ---------------------------------------------------------------------------------
            This pattern is not implemented. Please check the following:
               - There is a labeled set for the required pattern.
               - There is a second neural network subclass implemented for this pattern.
               - The pattern name is well written.
            -----------------------------------------------------------------------------
            '''
            print(message)
            sys.exit()
        else:
            # iterates through the generators directory, identifies the folders and enter in them
            for (dirpath, _dirnames, filenames) in os.walk(path):
                if dirpath.endswith('withpattern') or dirpath.endswith('nopattern'):
                    for filename in filenames:
                        if filename.endswith('.py'):
                            filepath = os.path.join(dirpath, filename)
                            yield filepath


    def set_leaves(self, ls_nodes):
        for node in ls_nodes:
            node.set_leaves()

    def set_vector(self, ls_nodes):
        df = pd.read_csv('initial_vector_representation.csv')
        for node in ls_nodes:
            node.set_vector(df)


def read_params(file):
    with open(file) as f:
        for line in f.readlines():
            words = line.split()
            if words[0] == 'folder':
                # Convert a string into a variable
                folder = words[2]
                #exec("%s = %d" % (words[0], words[2]))
            elif words[0] == 'pattern':
                pattern = words[2]
            elif words[0] == 'vector_size':
                vector_size = int(words[2])
            elif words[0] == 'learning_rate':
                learning_rate = float(words[2])
            elif words[0] == 'epoch':
                epoch = int(words[2])
            elif words[0] == 'momentum':
                momentum = float(words[2])
            elif words[0] == 'l2_penalty':
                l2_penalty = float(words[2])
               

    return folder, pattern, vector_size, learning_rate, momentum, l2_penalty, epoch


########################################

if __name__ == '__main__':

    folder, pattern, vector_size, learning_rate, momentum, l2_penalty, epoch = read_params('parameters.txt')

    vector_representation = Vector_representation(folder, pattern, vector_size, learning_rate, momentum, l2_penalty, epoch)
    vector_representation.vector_representation()