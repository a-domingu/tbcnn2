# TODO cambiar esto por el main cuando funcione
from main_tester import main
from utils import writer, remover
import os


# TODO asignar los valores que queramos para cada caso

vector_size_ls = [20]

learning_rate_ls = [0.1]

momentum_ls = [0.01]

# Second neural network parameters
feature_size_ls = [4]
epoch = 10
pooling = 'one-way pooling'
l2_penalty_ls = [0]




for vector_size in vector_size_ls:
    for learning_rate in learning_rate_ls:
        for momentum in momentum_ls:
            for learning_rate2 in [0.1]:
                for feature_size in feature_size_ls:
                    for l2_penalty in l2_penalty_ls:
                        message = f'''

########################################

The parameters we're using are the following:
vector_size = {vector_size}
learning_rate = {learning_rate}
momentum = {momentum}
learning_rate2 = {learning_rate2}
feature_size = {feature_size}
number of epochs for second neural network: {epoch}


                    '''
                    remover()
                    writer(message)
                    main(vector_size, learning_rate, momentum, learning_rate2,\
                         feature_size, epoch, pooling, l2_penalty)




