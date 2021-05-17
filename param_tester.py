from main import first_neural_network, second_neural_network
from utils import writer, remover
import os


# TODO asignar los valores que queramos para cada caso
# Folder path
path = os.path.join('sets100', 'generators')
 
# First neural network parameters
vector_size_ls = [30]
learning_rate_ls = [0.3]
momentum_ls = [0]
l2_penalty_ls = [0]
epoch_first = 45
 
# Second neural network parameters
feature_size_ls = [200, 300, 400]
learning_rate2_ls = [0.01, 0.001]
epoch = 30
pooling = 'one-way pooling'

# If exists a results.txt file, then we remove it
remover()

for vector_size in vector_size_ls:
    for learning_rate in learning_rate_ls:
        for momentum in momentum_ls:
            for l2_penalty in l2_penalty_ls:
                data_dict = first_neural_network(path, vector_size, learning_rate, momentum, l2_penalty, epoch_first)
                for learning_rate2 in learning_rate2_ls:
                    for feature_size in feature_size_ls:
                        message = f'''

########################################

The parameters we're using are the following:
vector_size = {vector_size}
learning_rate = {learning_rate}
momentum = {momentum}
l2_penalty = {l2_penalty}
number of epochs for first neural network: {epoch_first}
learning_rate2 = {learning_rate2}
feature_size = {feature_size}
number of epochs for second neural network: {epoch}
pooling method = {pooling}

                        '''
                        # We append the results in a results.txt file
                        writer(message)
                        second_neural_network(path, data_dict, vector_size, learning_rate2, feature_size, epoch, pooling)



