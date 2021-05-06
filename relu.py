import numpy as np


def relu(x):
    '''This is the RELU function ( max(x, 0) ) applied to a numpy vector, so it should apply the relu function to every element of the vector'''
    y = []
    for x_i in x:
        y_i = max(0, x_i)
        y.append(y_i)
    return np.array(y)