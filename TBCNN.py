from vector_representation import Vector_representation
from pattern_training import Pattern_training 
from initialize_vector_representation import Initialize_vector_representation
from parameters import folder, pattern, vector_size, feature_size, learning_rate, learning_rate2, epoch_first, epoch, batch, momentum, l2_penalty


def main(folder, pattern, vector_size, feature_size, learning_rate, learning_rate2, epoch_first, epoch, batch, momentum, l2_penalty):

    # We make an initial vector representation based on the type of node and/or some other features
    initial_vector_representation = Initialize_vector_representation(folder, pattern, vector_size)
    initial_vector_representation.initial_vector_representation()

    # We make the vector representation for all files
    vector_representation = Vector_representation(folder, pattern, vector_size, learning_rate, momentum, l2_penalty, epoch_first)
    vector_representation.vector_representation()

    # We train the model to detect the pattern
    pattern_training = Pattern_training(folder, pattern, vector_size, learning_rate2, feature_size, epoch, batch)
    pattern_training.pattern_training()
    

########################################

if __name__ == '__main__':

    main(folder, pattern, vector_size, feature_size, learning_rate, learning_rate2, epoch_first, epoch, batch, momentum, l2_penalty)