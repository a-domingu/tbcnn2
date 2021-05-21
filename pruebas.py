import numpy
import torch
import pickle
import torch.nn as nn
import pandas as pd

import node_object_creator 
from first_neural_network import First_neural_network
'''
x = torch.ones(10, 5, 5)
print(x)
y = torch.rand(5)
print(y)
y_1 = y.unsqueeze(0)
y_1 = torch.t(y_1)
#print(y_1)
y_1 = torch.stack((y, y, y, y, y, y, y, y, y, y), 0)
#print(y_1)
y_1 = y_1.unsqueeze(0)
y_1 = torch.transpose(y_1, 0, 1)
y_1 = torch.transpose(y_1, 1, 2)
print(y_1.size())
print(y_1)
y_2 = torch.rand(10,1,1)
print(y_2)
z_1 = y_2*x
print(z_1)
y_3 = torch.rand(10, 5, 1)
z = torch.matmul(x, y_3)
print(y_3)
print(z)
print(z.size())
'''
'''
y_1 = torch.transpose(y_1,0,1)
print(y_1.size())
y_1 = torch.transpose(y_1,1,1)
print(y_1.size())
'''
#z = torch.matmul(y_1,x)
#z = y_1*x
#z = torch.dot(y_1, x)
#print(x+z_1)
'''
x_1 = torch.ones(5,5)
z_1 = x_1*y)
print(z_1)

x = torch.ones(10, 1, 1)
print(x)
y = torch.rand(5, 5)
print(y)
#z = torch.matmul(x,y)
z = x * y
print(z)
print('Shape: ', z.shape)
z = torch.sum(z, 0)
print(z)
print('Shape: ', z.shape)
'''
'''
x = torch.tensor([True, False, False, False])
print(x.sum())
y = torch.tensor([False, False, False, False])
print(y.sum())
'''
# Multiplicar un tensor 3D con un tamaño (n x 1 x 1) y un tensor 2D con un tamaño (30 x 30) y obtener un tensor
# 3D con un tamaño (n x 30 x 30)
'''
x = torch.rand(30, dtype=torch.float32)
print(x.shape)
y = torch.squeeze(torch.distributions.Uniform(-1, +1).sample((30, 1)), 1)
print(y.shape)
'''
'''
x = torch.ones(10, 5, 5)
print(x)
y = torch.rand(10, 5, 1)
print(y)
z = torch.matmul(x,y)
#z = x*y
print(z)
#z2 = z + z
#print(z2)
'''
'''
softmax = nn.Sigmoid()
criterion = nn.BCELoss()
criterion2 = nn.BCEWithLogitsLoss()
targets = torch.tensor([1.0])
output = torch.tensor([numpy.nan])
outputs = softmax(output)
print(outputs)
try:
    loss = criterion(outputs, targets)
except RuntimeError:
    loss = torch.tensor([numpy.nan])
loss2 = criterion2(output, targets)
print(loss)
'''
'''
def main():
    tree = 'test\\generators\\prueba.py'

    main_node = node_object_creator.node_object_creator(tree)
    
    ls_nodes = main_node.descendants()
    set_leaves(ls_nodes)

    # Initializing vector embeddings
    set_vector(ls_nodes)

    print('Initial vectors: ')

    for node in ls_nodes:
        print(node.vector)
        break

    for node in main_node.descendants():
        print(node.vector)
        print('####################')
        break

    # Calculate the vector representation for each node
    vector_representation = First_neural_network(ls_nodes, 30, 0.1, 0, 0, 5)
    ls_nodes, w_l_code, w_r_code, b_code = vector_representation.vector_representation()

    print('After first neural network: ')

    for node in ls_nodes:
        print(node.vector)
        break

    for node in main_node.descendants():
        print(node.vector)
        print('####################')
        break


    filename = tree + '.txt'
    params = [ls_nodes, w_l_code, w_r_code, b_code]

    with open(filename, 'wb') as f:
        pickle.dump(params, f)

    print(f"Vector rep. of file: {tree}")

    with open(filename, 'rb') as f:
        params_2 = pickle.load(f) 

    ls_nodes_2 = params_2[0]

    print('After save vectors with pickle: ')
    
    for node in ls_nodes_2:
        print(node.vector)
        break

    main_node_2 = ls_nodes_2[0]
    for node in main_node_2.descendants():
        print(node.vector)
        print('#################### \n')
        break

    
def set_leaves(ls_nodes):
    for node in ls_nodes:
        node.set_leaves()

def set_vector(ls_nodes):
    df = pd.read_csv('initial_vector_representation.csv')
    for node in ls_nodes:
        node.set_vector(df)

if __name__ == '__main__':
    main()
'''