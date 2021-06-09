import os
import click

from vector_representation import Vector_representation
from pattern_training import Pattern_training 
from initialize_vector_representation import Initialize_vector_representation


@click.command()
@click.argument('pattern', required = True, nargs=1, type=str)
@click.option('--vector_size', required = False, type=int, default = 30, help='Choose the size of the vector representation')  
@click.option('--feature_size', required = False, type=int, default = 100, help='Choose the number of features detectors') 
@click.option('--learning_rate', required = False, type=int, default = 0.3, help='Choose the learning rate for the vector representation')
@click.option('--learning_rate2', required = False, type=int, default = 0.001, help='Choose the learning rate for the TBCNN')
@click.option('--epoch_first', required = False, type=int, default = 1, help='Choose the number of epochs for the vector representation') 
@click.option('--epoch', required = False, type=int, default = 30, help='Choose the number of epochs for the TBCNN') 
@click.option('--batch', required = False, type=int, default = 64, help='Choose the batch size') 
@click.option('--momentum', required = False, type=int, default = 0, help='Parameter (epsilon) used in the SGD with momentum algorithm') 
@click.option('--l2_penalty', required = False, type=int, default = 0, help='Hyperparameter that strikes the balance between coding error and l2 penalty')
@click.option('--initial_vector_representation', required = False, is_flag = True, help='Make an initial vector representation based on the type of node')
def main(pattern, vector_size, feature_size, learning_rate, learning_rate2, epoch_first, epoch, batch, momentum, l2_penalty, initial_vector_representation):

    # We make an initial vector representation based on the type of node and/or some other features
    if initial_vector_representation:
        initial_vector_representation = Initialize_vector_representation(pattern, vector_size)
        initial_vector_representation.initial_vector_representation()

    # We make the vector representation for all files
    vector_representation = Vector_representation(pattern, vector_size, learning_rate, momentum, l2_penalty, epoch_first)
    vector_representation.vector_representation()

    # We train the model to detect the pattern
    pattern_training = Pattern_training(pattern, vector_size, learning_rate2, feature_size, epoch, batch)
    pattern_training.pattern_training()

########################################

if __name__ == '__main__':
    main()