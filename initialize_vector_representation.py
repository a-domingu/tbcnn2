import os
import sys
import pandas as pd

from utils.node_object_creator import *
from first_neural_network.embeddings import Embedding
from parameters import folder, pattern, vector_size


class Initialize_vector_representation():

    def __init__(self, folder, pattern, vector_size):
        self.folder = folder
        self.pattern = pattern
        self.vector_size = vector_size


    def initial_vector_representation(self):
        # Training the first neural network
        vectors_dict = self.first_neural_network()
        #save_files(ls_nodes)
        self.save_vectors(vectors_dict)


    def save_vectors(self, vectors_dict):
        df = pd.DataFrame.from_dict(vectors_dict)
        df.to_csv('initial_vector_representation.csv')


    def first_neural_network(self):
        # we create the data dict with all the information about vector representation
        data_dict = self.first_neural_network_dict_creation()
        # We now do the first neural network (vector representation) for every file:
        data_dict = self.vector_representation_all_files(data_dict)
        return data_dict


    def first_neural_network_dict_creation(self):
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
            # we create the data dict with all the information about vector representation
            data_dict = {}
            # iterates through the generators directory, identifies the folders and enter in them
            for (dirpath, _dirnames, filenames) in os.walk(path):
                if dirpath.endswith('withpattern') or dirpath.endswith('nopattern'):
                    for filename in filenames:
                        if filename.endswith('.py'):
                            filepath = os.path.join(dirpath, filename)
                            data_dict[filepath] = None

            return data_dict


    def vector_representation_all_files(self, data_dict):
        ls_nodes_all = []
        for tree in data_dict:
        
            # convert its nodes into the Node class we have, and assign their attributes
            main_node = node_object_creator(tree)
        
            for node in main_node.descendants():
                ls_nodes_all.append(node)

        # Initializing vector embeddings
        embed = Embedding(self.vector_size, ls_nodes_all)
        dc = embed.node_embedding()
        return dc



########################################

if __name__ == '__main__':

    initial_vector_representation = Initialize_vector_representation(folder, pattern, vector_size)
    initial_vector_representation.initial_vector_representation()