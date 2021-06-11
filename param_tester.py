from vector_representation import Vector_representation
from pattern_training import Pattern_training
from utils.utils import writer, remover
import os


# Folder path
pattern = 'generator'
 
# First neural network parameters
vector_size_ls = [30]
learning_rate_ls = [0.3]
momentum_ls = [0]
l2_penalty_ls = [0]
epoch_first = 30
 
# Second neural network parameters
feature_size_ls = [50, 100]
learning_rate2_ls = [0.001]
epoch = 45
batch_size = 20
pooling = 'one-way pooling'

# If exists a results.txt file, then we remove it
remover()

for vector_size in vector_size_ls:
    for learning_rate in learning_rate_ls:
        for momentum in momentum_ls:
            for l2_penalty in l2_penalty_ls:
                first_neural_network = Vector_representation(pattern, vector_size, learning_rate, momentum, l2_penalty, epoch_first)
                first_neural_network.vector_representation()
                for learning_rate2 in learning_rate2_ls:
                    for feature_size in feature_size_ls:
                        message = f'''

########################################

The parameters we're using are the following:
pattern = {pattern}
vector_size = {vector_size}
learning_rate = {learning_rate}
momentum = {momentum}
l2_penalty = {l2_penalty}
number of epochs for first neural network: {epoch_first}
learning_rate2 = {learning_rate2}
feature_size = {feature_size}
number of epochs for second neural network: {epoch}
batch_size = {batch_size}
pooling method = {pooling}

                        '''
                        # We append the results in a results.txt file
                        writer(message)
                        second_neural_network = Pattern_training(pattern, vector_size, learning_rate2, feature_size, epoch, batch_size)
                        second_neural_network.pattern_training()



