import random
import torch as torch
import torch.nn as nn
import torch.nn.functional as F

from first_neural_network.node import Node

class First_neural_network():
    '''
    In this class we update vec(·), where vec(·) is the feature representation of a node in the AST.
    We use the stochastic gradient descent with momentum algorithm of the 
    "Building Program Vector Representations for Deep Learning" report.
    First we compute the cost function J by using the coding criterion d and then we applied the back
    propagation algorithm
    Inputs:
    ls_nodes [list <class Node>]: list with all nodes in the AST
    dict_ast_to_Node[dict[ast_object] = <class Node>]: dictionary that relates class ast objects to class Node objects
    features_size [int]: Vector embedding size
    learning_rate [int]: learning rate parameter 'alpha' used in the SGD algorithm
    momentum [int]: momentum parameter 'epsilon' used in the SGD with momentum algorithm
    l2_penalty [int]: hyperparameter that strikes the balance between coding error and l2 penalty.
    
    Output:
    ls_nodes [list <class Node>]: We update vector embedding of all nodes
    w_l [matrix[features_size x features_size]]: left weight matrix used as parameter
    w_r [matrix[features_size x features_size]]: right weight matrix used as parameter
    b [array[features_size]]: bias term
    '''

    def __init__(self, ls_nodes, features_size, learning_rate, momentum, l2_penalty, epoch):
        self.ls = ls_nodes
        self.features_size = features_size
        self.alpha = learning_rate
        self.epsilon = momentum
        self.l2_penalty = l2_penalty
        self.total_epochs = epoch
        self.w_l = torch.distributions.Uniform(-1, +1).sample((self.features_size, self.features_size)).requires_grad_()
        self.w_r = torch.distributions.Uniform(-1, +1).sample((self.features_size, self.features_size)).requires_grad_()
        self.b = torch.squeeze(torch.distributions.Uniform(-1, +1).sample((self.features_size, 1))).requires_grad_()
        self.vector_matrix = None
        self.vector_p = None
        self.l_vector = None
        self.w_l_params = None
        self.w_r_params = None


    def train(self):
        ### SGD
        # params is a tensor with vectors (p -> node.vector and node childs c1,..,cN -> node_list), w_r, w_l and b
        params = [node.vector for node in self.ls]
        params.append(self.w_l)
        params.append(self.w_r)
        params.append(self.b)
        # Construct the optimizer
        # Stochastic gradient descent with momentum algorithm
        optimizer = torch.optim.SGD(params, lr = self.alpha, momentum = self.epsilon)

        for step in range(self.total_epochs):
            # Training loop (forward step)
            output = self.training_iterations()

            # Computes the cost function (loss)
            loss = self.cost_function_calculation(output)

            # Calculates the derivative
            loss.backward() #self.w_l.grad = dloss/dself.w_l
            
            # Update parameters
            optimizer.step() #self.w_l = self.w_l - lr * self.w_l.grad

            # Zero gradients
            optimizer.zero_grad()

        for node in self.ls:
            node.vector.detach()

        return self.ls, self.w_l.detach(), self.w_r.detach(), self.b.detach()


    # We applied the coding criterion for each non-leaf node p in AST
    def training_iterations(self):
        sum_error_function = torch.tensor([0])
        for node in self.ls:
            if len(node.children) > 0:
                # Generates training sample and computes d 
                self.param_initializer(node)
                d = self.compute_distance(node)
                # Generates a negative sample and computes d_c
                self.negative_sample_d_c(node)
                d_c = self.compute_distance(node)
                # Computes the error function J(d,d_c) for each node and computes the sum
                sum_error_function = sum_error_function + self.error_function(d_c, d)       
        return sum_error_function

    
    def param_initializer(self, node):
        # TODO esto no es necesario realizarlo en cada epoch
        # We save the vector p
        self.vector_p = node.vector
        # We create a list with all vectors and a list with all l_i values
        vectors = []
        vector_l = []
        # Parameters used to calculate the weight matrix for each node
        n = len(node.children)
        w_l_list = []
        w_r_list = []
        i = 1
        # child is an AST object and we convert the AST object to a Node object
        for child in node.children:
            vectors.append(child.vector)
            # TODO solucionar esto de aqui abajo (obtener las hojas)
            vector_l.append((len(child.leaves)/len (node.leaves)))
            if n>1:
                w_l_list.append((n-i)/(n-1))
                w_r_list.append((i-1)/(n-1))
                i += 1
        # We create a matrix with all the vectors
        self.vector_matrix = torch.stack(tuple(vectors), 0)
        # We create a vector with all the l_i values
        self.l_vector = torch.tensor(vector_l)
        # We create a tensor with the parameters associated to the left matrix
        self.w_l_params = torch.tensor(w_l_list)
        # We create a tensor with the parameters associated to the right matrix
        self.w_r_params = torch.tensor(w_r_list)
        # Reshape the matrices and vectors and create 3D tensors
        self.reshape_matrices_and_vectors()

    
    # Reshape the matrices and vectors and create 3D tensors
    def reshape_matrices_and_vectors(self):
        # We create a 3D tensor for the vector matrix: shape(nb_nodes, 30, 1)
        self.vector_matrix = torch.unsqueeze(self.vector_matrix, 2)

        # We create a 3D tensor for the vector with all l_i values: shape(nb_nodes, 1, 1)
        self.l_vector = torch.unsqueeze(self.l_vector, 1)
        self.l_vector = torch.unsqueeze(self.l_vector, 1)

        # We create a 3D tensor for the parameters associated to the left matrix: shape(nb_nodes, 1, 1)
        self.w_l_params = torch.unsqueeze(self.w_l_params, 1)
        self.w_l_params = torch.unsqueeze(self.w_l_params, 1)

        # We create a 3D tensor for the parameters associated to the right matrix: shape(nb_nodes, 1, 1)
        self.w_r_params = torch.unsqueeze(self.w_r_params, 1)
        self.w_r_params = torch.unsqueeze(self.w_r_params, 1)

    
    def negative_sample_d_c(self, node):
        # We choose a Node class that cames from a ls_nodes    
        symbol = random.choice(self.ls)
        # We substitutes randomly a vector with a different vector
        index = random.randint(0, self.vector_matrix.shape[0])
        if index == 0:
            self.vector_p = symbol.vector
        else:
            vector = torch.unsqueeze(symbol.vector, 1)
            vector = torch.unsqueeze(vector, 0)
            index = torch.tensor([index])
            self.vector_matrix = torch.index_copy(self.vector_matrix, 0, index-1, vector)
            # We replace the l_i associted to the new symbol
            leaves_nodes = torch.tensor([len(symbol.leaves)])
            leaves_nodes = torch.unsqueeze(leaves_nodes, 1)
            leaves_nodes = torch.unsqueeze(leaves_nodes, 1)
            self.l_vector = torch.index_copy(self.l_vector, 0, index-1, leaves_nodes.float()) 


    # Compute the cost function (function objective)
    def cost_function_calculation(self, sum_J):
        first_term = (1/len(self.ls)*sum_J)
        # Norms calculations
        norm_w_l = torch.norm(self.w_l, p='fro')
        squared_norm_w_l = norm_w_l * norm_w_l
        norm_w_r = torch.norm(self.w_r, p='fro')
        squared_norm_w_r = norm_w_r * norm_w_r
        # Second term calculation(Revisar el parametro lambda del paper!!!)
        second_term = (self.l2_penalty/(2*2*self.features_size*self.features_size))*(squared_norm_w_l + squared_norm_w_r)
        return first_term + second_term


    # Calculate the error function J(d,d_c)
    def error_function(self, d_c, d):
        margin = torch.tensor([1])
        error_function = margin + d - d_c
        return max(torch.tensor([0]), error_function)


    # Calculate the square of the Euclidean distance between the real vector and the target value.
    def compute_distance(self, node):
        # Calculate the target value
        if self.vector_matrix.shape[0] > 1:
            calculated_vector = self.calculate_vector(node)
        elif self.vector_matrix.shape[0] == 1:
            calculated_vector = self.calculate_vector_special_case()

        # Calculate the square of the Euclidean distance, d
        diff_vector = self.vector_p - calculated_vector
        euclidean_distance = torch.norm(diff_vector, p=2)
        d = euclidean_distance * euclidean_distance
        return d


    # Calculate the target value
    def calculate_vector(self, node):
        # Calculate the weighted matrix for each node as a linear combination of matrices w_l and w_r
        # We compute the weighted matrix: shape(nb_nodes, 30, 30)
        weighted_matrix = self.l_vector*((self.w_l_params*self.w_l)+(self.w_r_params*self.w_r))

        # Sum the weighted values over vec(·)
        final_matrix = torch.matmul(weighted_matrix, self.vector_matrix)
        final_vector = torch.sum(final_matrix, 0)
        final_vector = torch.squeeze(final_vector, 1)

        return F.relu(final_vector + self.b, inplace=False)


    # Calculate the weighted matrix for a node with only one child
    def calculate_vector_special_case(self):
        matrix = ((1/2)*self.w_l) + ((1/2)*self.w_r)
        vector = self.vector_matrix
        vector = torch.squeeze(vector, 2)
        vector = torch.squeeze(vector, 0)
        final_vector = torch.matmul(matrix, vector) + self.b
        return F.relu(final_vector, inplace=False)
