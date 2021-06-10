import main_first_neural_network, main_second_neural_network, initialize_vector_representation
from utils import writer, remover
import os

if __name__ == '__main__':


    # Folder path
    path = os.path.join('sets200', 'wrappers')
    
    # First neural network parameters
    vector_size_ls = [30]
    learning_rate_ls = [0.3]
    momentum_ls = [0]
    l2_penalty_ls = [0]
    epoch_first = 1
    
    # Second neural network parameters
    feature_size_ls = [200]
    learning_rate2_ls = [0.001]
    epoch = 1
    batch_size = 20
    pooling = 'one-way pooling'

    # If exists a results.txt file, then we remove it
    remover()




    for vector_size in vector_size_ls:
        initialize_vector_representation.main(path, vector_size)
        for learning_rate in learning_rate_ls:
            for momentum in momentum_ls:
                for l2_penalty in l2_penalty_ls:
                    main_first_neural_network.main(path, vector_size, learning_rate, momentum, l2_penalty, epoch_first)
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
    batch size = {batch_size}
    pooling method = {pooling}

                            '''
                            # We append the results in a results.txt file
                            writer(message)
                            main_second_neural_network.main(path, vector_size, learning_rate2, feature_size, epoch, pooling, batch_size)



