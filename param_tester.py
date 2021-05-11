from main import main
from utils import writer, remover
import os


# TODO asignar los valores que queramos para cada caso
# First neural network parameters
vector_size_ls = [30]
learning_rate_ls = [0.3]
momentum_ls = [0]
l2_penalty_ls = [0]
epoch_first = 45

# Second neural network parameters
feature_size_ls = [100, 200, 300]
learning_rate2_ls = [0.1, 0.01, 0.001]
epoch = 40
pooling = 'one-way pooling'

# If exists a results.txt file, then we remove it
remover()

for vector_size in vector_size_ls:
    for learning_rate in learning_rate_ls:
        for momentum in momentum_ls:
            for learning_rate2 in learning_rate2_ls:
                for feature_size in feature_size_ls:
                    for l2_penalty in l2_penalty_ls:
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
                        main(vector_size, learning_rate, momentum, epoch_first, learning_rate2,\
                            feature_size, epoch, pooling, l2_penalty)




