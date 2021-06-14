from get_input import Get_input
from parameters import *
import pickle



with open('vector_representation\\brew.py.txt', 'br') as f:
    ls_nodes = pickle.load(f)[0]



get_input = Get_input(ls_nodes, vector_size, 'vector_representation\\brew.py.txt')

get_input.get_input()
