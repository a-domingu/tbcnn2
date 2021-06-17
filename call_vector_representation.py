from parameters import *
from vector_representation import Vector_representation

x = Vector_representation(folder, pattern, vector_size=vector_size, learning_rate=learning_rate,momentum = 0,l2_penalty= 0,epoch=1)
x.vector_representation()