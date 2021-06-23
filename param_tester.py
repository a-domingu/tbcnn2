from vector_representation import Vector_representation
from pattern_training import Pattern_training
from initialize_vector_representation import Initialize_vector_representation
from utils.utils import writer, remover
from parameters import folder
import os


if __name__ == '__main__':

    # Folder path
    folder = 'sets'
    pattern = 'wrapper'
    
    # First neural network parameters
    vector_size_ls = [30, 100]
    learning_rate_ls = [0.001, 0.01]
    momentum_ls = [0]
    l2_penalty_ls = [0]
    epoch_first_ls = [5,10]
    
    # Second neural network parameters
    feature_size_ls = [100]
    learning_rate2_ls = [0.001]
    epoch = 30
    batch_size = 20
    pooling = 'one-way pooling'

    # If exists a results.txt file, then we remove it
    remover()

    for vector_size in vector_size_ls:
        initialize_vector_representation = Initialize_vector_representation(folder, pattern, vector_size)
        initialize_vector_representation.initial_vector_representation()
        for learning_rate in learning_rate_ls:
            for momentum in momentum_ls:
                for l2_penalty in l2_penalty_ls:
                    for epoch_first in epoch_first_ls:
                        first_neural_network = Vector_representation(folder, pattern, vector_size, learning_rate, momentum, l2_penalty, epoch_first)
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
                                second_neural_network = Pattern_training(folder, pattern, vector_size, learning_rate2, feature_size, epoch, batch_size)
                                second_neural_network.pattern_training()



