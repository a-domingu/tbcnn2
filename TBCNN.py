from vector_representation import Vector_representation
from pattern_training import Pattern_training 
from initialize_vector_representation import Initialize_vector_representation


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
            elif words[0] == 'feature_size':
                feature_size = int(words[2]) 
            elif words[0] == 'learning_rate':
                learning_rate = float(words[2])
            elif words[0] == 'learning_rate2':
                learning_rate2 = float(words[2]) 
            elif words[0] == 'epoch_first':
                epoch_first = int(words[2])
            elif words[0] == 'epoch':
                epoch = int(words[2])
            elif words[0] == 'batch':
                batch = int(words[2])
            elif words[0] == 'momentum':
                momentum = float(words[2])
            elif words[0] == 'l2_penalty':
                l2_penalty = float(words[2])
            else:
                print('The parameter: ', words[0], 'is not implemented or well written')          

    return folder, pattern, vector_size, feature_size, learning_rate, learning_rate2, epoch_first, epoch, batch, momentum, l2_penalty

########################################

if __name__ == '__main__':

    folder, pattern, vector_size, feature_size, learning_rate, learning_rate2, epoch_first, epoch, batch, momentum, l2_penalty = read_params('parameters.txt')
    main(folder, pattern, vector_size, feature_size, learning_rate, learning_rate2, epoch_first, epoch, batch, momentum, l2_penalty)