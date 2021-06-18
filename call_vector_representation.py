from parameters import *
from vector_representation import Vector_representation

x = Vector_representation(folder, pattern, vector_size = vector_size, learning_rate = learning_rate, momentum = momentum, l2_penalty = l2_penalty, epoch_first = epoch_first)
x.vector_representation()