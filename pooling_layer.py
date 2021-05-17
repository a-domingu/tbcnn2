import torch

class Pooling_layer():
    
    '''
    This class will receive a list of nodes (of 'Node' type), from which we'll take their node.y vector,
    and apply the max pool function. This function will simply return the maximum element of node.y 
    (infinity norm), and we'll save it as an atribute of each node
    '''

    def __init__(self):
        self.ls = []


    def pooling_layer(self, ls_nodes):
        # Initialize the node list
        self.ls = ls_nodes

        ls_tensors = self.create_ls_tensors(self.ls)
        matrix = self.create_matrix(ls_tensors)
        #print('tree_tensor: \n', matrix)
        pooled_tensor, _indices = self.one_way_pooling(matrix)
        return pooled_tensor

    def create_ls_tensors(self, ls_nodes):
        ls_tensors = []
        for node in ls_nodes:
            ls_tensors.append(node.y)
        return ls_tensors
    
    def create_matrix(self, ls):
        matrix = torch.stack(ls)
        return matrix
    
    def one_way_pooling(self, tensor):
        pool_tensor = torch.max(tensor, dim = 0)
        return pool_tensor
        





