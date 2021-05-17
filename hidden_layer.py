import torch


class Hidden_layer():
    
    def __init__(self):
        self.input = []
        self.w = None
        self.b = None
        # The size of n is based on the dynamic pooling method.
        # In one-way pooling the size of n is equal to the output_size / feature_detectors
        # In three-way pooling the size of n is equal to 3
        self.n = 3


    def hidden_layer(self, vector, w, b):
        # Initialize matrix and vector
        self.w = w
        self.b = b
        # Initialize the node list and the vector
        self.input = vector
        output = self.get_output()
        return output

    def get_output(self):
        output = torch.matmul(self.w,self.input) + self.b
        return output