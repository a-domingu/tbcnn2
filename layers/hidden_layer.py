import torch

class Hidden_layer():
    
    def __init__(self, feature_size):
        self.input = []
        self.feature_size = feature_size
        self.w = torch.squeeze(torch.distributions.Uniform(-1, +1).sample((self.feature_size, 1))).requires_grad_()
        self.b = torch.rand(1, requires_grad = True)
        # The size of n is based on the dynamic pooling method.
        # In one-way pooling the size of n is equal to the output_size / feature_detectors
        # In three-way pooling the size of n is equal to 3
        self.n = 3


    def hidden_layer(self, vector):
        # Initialize the node list and the vector
        self.input = vector
        output = self.get_output()
        return output

    def initialize_matrices_and_bias(self):
        return self.w, self.b

    def set_matrices_and_vias(self, w, b):
        self.w, self.b = w, b

    def get_output(self):
        output = torch.matmul(self.w,self.input) + self.b
        return output