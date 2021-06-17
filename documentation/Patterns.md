# New patterns

Before reading this document, please read carefully the instructions first.

## How to train the neural network to detect new patterns?

To train the neural network you should have a data set and a test set for the new pattern. Also, you should create the following subclasses: a second neural network subclass, a pattern test subclass and a pattern detector subclass for this new pattern. 

**Note**: The structure of the data set and test set should follow the structure explained in the file `Instructions.md` and section `Data set`.


To create a new second neural network subclass you should create a new file in the folder **second_neural_network**. The name of the file should have the following structure: 'pattern'+'_second_neural_network'. For example, if you want to create a `decorator` subclass, the file name must be `decorator_second_neural_network.py`. Also, the name of the class should have the same structure. In this case, the class name should be `Decorator_second_neural_network`.

**Note**: You can copy the content of the `generator_second_neural_network.py` and change the class name. Also, if you want to add more changes, you can check the `Second neural network abstract class` section to get more information.


To create a new pattern test subclass you should create a new file in the folder **pattern_accuracy_test**. The name of the file should have the following structure: 'pattern'+'_test'. For example, if you want to create a `decorator` subclass, the file name must be `decorator_test.py`. Also, the name of the class should have the same structure. In this case, the class name should be `Decorator_test`.

**Note**: You can copy the content of the `generator_test.py` and change the class name. Also, if you want to add more changes, you can check the `Pattern test abstract class` section to get more information.


To create a new pattern test subclass you should create a new file in the folder **pattern_detection**. The name of the file should have the following structure: 'pattern'+'_detector'. For example, if you want to create a `decorator` subclass, the file name must be `decorator_detector.py`. Also, the name of the class should have the same structure. In this case, the class name should be `Decorator_detector`.

**Note**: You can copy the content of the `generator_detector.py` and change the class name. Also, if you want to add more changes, you can check the `Pattern detector abstract class` section to get more information.


## Second neural network abstract class

The `second neural network` is an abstract class that allows us to train the CNN to detect patterns based on the data set. All the second neural network subclasses has multiple common functions. However, each pattern has three particular functions: `matrices and layers initialization`, `layers` and `save`.

In the `matrices and layers initialization` function, you can choose the layers you want to use as well as the number of convolutional layers. Also, the function should return all the matrices and vectors used to detect patterns. For example, if you want to use the `coding layer` to detect the new patter you just simply add the following code:

```
def matrices_and_layers_initialization(self):
    # Initialize the layers
    self.cod = Coding_layer(self.vector_size)
    self.conv = Convolutional_layer(self.vector_size, features_size = self.feature_size)
    self.hidden = Hidden_layer(self.feature_size)
    self.pooling = Pooling_layer()

    # Initialize matrices and bias
    self.w_comb1, self.w_comb2 = self.cod.initialize_matrices_and_bias()
    self.w_t, self.w_l, self.w_r, self.b_conv = self.conv.initialize_matrices_and_bias()
    self.w_hidden, self.b_hidden = self.hidden.initialize_matrices_and_bias()

    params = [self.w_comb1, self.w_comb2, self.w_t, self.w_l, self.w_r, self.b_conv, self.w_hidden, self.b_hidden]

    return params
```

The `layers` function should have all the layers you want to use and you should write them in order. For example, if you want to apply the `convolutional layer` multiple times you should first initialize them in the `matrices and layers initialization` function and then add them in the `layers` function. The code will be as below:

```
def matrices_and_layers_initialization(self):
    # Initialize the layers
    self.conv1 = Convolutional_layer(self.vector_size, features_size = self.feature_size)
    self.conv2 = Convolutional_layer(self.feature_size, features_size = self.feature_size)
    self.hidden = Hidden_layer(self.feature_size)
    self.pooling = Pooling_layer()

    # Initialize matrices and bias
    self.w_t_1, self.w_l_1, self.w_r_1, self.b_conv_1 = self.conv1.initialize_matrices_and_bias()
    self.w_t_2, self.w_l_2, self.w_r_2, self.b_conv_2 = self.conv2.initialize_matrices_and_bias()
    self.w_hidden, self.b_hidden = self.hidden.initialize_matrices_and_bias()

    params = [self.w_t_1, self.w_l_1, self.w_r_1, self.b_conv_1, self.w_t_2, self.w_l_2, self.w_r_2, self.b_conv_2, self.w_hidden, self.b_hidden]

    return params


def layers(self, vector_representation_params):
    # Parameters of the first neural network
    ls_nodes, w_l_code, w_r_code, b_code = vector_representation_params
    # Convolutional layer
    ls_nodes = self.conv1.convolutional_layer(ls_nodes)
    ls_nodes = self.conv2.convolutional_layer(ls_nodes)
    # Pooling layer
    vector = self.pooling.pooling_layer(ls_nodes)
    # Hidden layer
    output = self.hidden.hidden_layer(vector)

    return output
```

The `save` function will save all the trained matrices and vectors (in a .csv) in a subfolder of the `params` folder. You should save all the matrices and vectors that are returned in the `matrices and layers initialization` function.


## Pattern test abstract class

The `pattern test` is an abstract class that allows us to test the accuracy of the CNN to detect patterns in code. All the pattern test subclasses has multiple common functions. However, each pattern has three particular functions: `set features size`, `load matrices and vectors` and `second neural network`.

The `set features size` function uses the first trained matrix of the convolutional layer (`w_t.csv`) to set the feature size. For this reason, if you change the name of this file, you should change its name in the `set features size` function. For example, if you apply the `convolutional layer` multiple times (previous example), you should write the following:

```
def set_feature_size(self):
    df = pd.read_csv(os.path.join('params', self.pattern, 'w_t_1.csv'))
    feature_size = len(df[df.columns[0]])

    return feature_size
```

The `load matrices and vectors` function reads all the trained matrices and vectors that are necessary to detect patterns in code. You should load all the matrices and vectors that are in the `pattern` folder, which is a subfolder of the `params` folder. You also need to set each matrix and each vector in its respective layer. For example, if you used the `coding layer` to train the CNN as above, you just simply add the following code:

def load_matrices_and_vectors(self):
    '''Load all the trained parameters from a csv file'''
    directory = os.path.join('params', self.pattern)
    if not os.path.exists(directory):
        os.mkdir(directory)

    # Coding layer
    w_comb1 = numpy.genfromtxt(os.path.join(directory, "w_comb1.csv"), delimiter = ",")
    w_comb1 = torch.tensor(w_comb1, dtype=torch.float32)

    w_comb2 = numpy.genfromtxt(os.path.join(directory, "w_comb2.csv"), delimiter = ",")
    w_comb2 = torch.tensor(w_comb2, dtype=torch.float32)

    self.cod.set_matrices_and_vias(w_comb1, w_comb2)

    # Convolutional layer
    w_t = numpy.genfromtxt(os.path.join(directory, "w_t.csv"), delimiter = ",")
    w_t = torch.tensor(w_t, dtype=torch.float32)

    w_r = numpy.genfromtxt(os.path.join(directory, "w_r.csv"), delimiter = ",")
    w_r = torch.tensor(w_r, dtype=torch.float32)

    w_l = numpy.genfromtxt(os.path.join(directory, "w_l.csv"), delimiter = ",")
    w_l = torch.tensor(w_l, dtype=torch.float32)

    b_conv = numpy.genfromtxt(os.path.join(directory, "b_conv.csv"), delimiter = ",")
    b_conv = torch.tensor(b_conv, dtype=torch.float32)

    self.conv.set_matrices_and_vias(w_t, w_l, w_r, b_conv)

    # Hidden layer
    w_hidden = numpy.genfromtxt(os.path.join(directory, "w_hidden.csv"), delimiter = ",")
    w_hidden = torch.tensor(w_hidden, dtype=torch.float32)

    b_hidden = numpy.genfromtxt(os.path.join(directory, "b_hidden.csv"), delimiter = ",")
    b_hidden = torch.tensor(b_hidden, dtype=torch.float32)

    self.hidden.set_matrices_and_vias(w_hidden, b_hidden)

The `second neural network` should have the same layers in the same order as the layers used to train the CNN. For example, if we apply the `convolutional layer` multiple times (as mention above), we should write the following:

```
def second_neural_network(self, vector_representation_params):
    # Parameters of the first neural network
    ls_nodes, w_l_code, w_r_code, b_code = vector_representation_params
    # Convolutional layer
    ls_nodes = self.conv1.convolutional_layer(ls_nodes)
    ls_nodes = self.conv2.convolutional_layer(ls_nodes)
    # Pooling layer
    vector = self.pooling.pooling_layer(ls_nodes)
    # Hidden layer
    output = self.hidden.hidden_layer(vector)

    return output
```


## Pattern detector abstract class

The `pattern detector` is an abstract class that allows us to check if a local file (or folder) or a GitHub repository constains the required pattern. All the pattern detecto subclasses has multiple common functions. However, each pattern has three particular functions: `set features size`, `load matrices and vectors` and `second neural network`.

The `set features size` function uses the first trained matrix of the convolutional layer (`w_t.csv`) to set the feature size. For this reason, if you change the name of this file, you should change its name in the `set features size` function. For example, if you apply the `convolutional layer` multiple times (previous example), you should write the following:

```
def set_feature_size(self):
    df = pd.read_csv(os.path.join('params', self.pattern, 'w_t_1.csv'))
    feature_size = len(df[df.columns[0]])

    return feature_size
```

The `load matrices and vectors` function reads all the trained matrices and vectors that are necessary to detect patterns in code. You should load all the matrices and vectors that are in the `pattern` folder, which is a subfolder of the `params` folder. You also need to set each matrix and each vector in its respective layer. For example, if you used the `coding layer` to train the CNN as above, you just simply add the following code:

def load_matrices_and_vectors(self):
    '''Load all the trained parameters from a csv file'''
    directory = os.path.join('params', self.pattern)
    if not os.path.exists(directory):
        os.mkdir(directory)

    # Coding layer
    w_comb1 = numpy.genfromtxt(os.path.join(directory, "w_comb1.csv"), delimiter = ",")
    w_comb1 = torch.tensor(w_comb1, dtype=torch.float32)

    w_comb2 = numpy.genfromtxt(os.path.join(directory, "w_comb2.csv"), delimiter = ",")
    w_comb2 = torch.tensor(w_comb2, dtype=torch.float32)

    self.cod.set_matrices_and_vias(w_comb1, w_comb2)

    # Convolutional layer
    w_t = numpy.genfromtxt(os.path.join(directory, "w_t.csv"), delimiter = ",")
    w_t = torch.tensor(w_t, dtype=torch.float32)

    w_r = numpy.genfromtxt(os.path.join(directory, "w_r.csv"), delimiter = ",")
    w_r = torch.tensor(w_r, dtype=torch.float32)

    w_l = numpy.genfromtxt(os.path.join(directory, "w_l.csv"), delimiter = ",")
    w_l = torch.tensor(w_l, dtype=torch.float32)

    b_conv = numpy.genfromtxt(os.path.join(directory, "b_conv.csv"), delimiter = ",")
    b_conv = torch.tensor(b_conv, dtype=torch.float32)

    self.conv.set_matrices_and_vias(w_t, w_l, w_r, b_conv)

    # Hidden layer
    w_hidden = numpy.genfromtxt(os.path.join(directory, "w_hidden.csv"), delimiter = ",")
    w_hidden = torch.tensor(w_hidden, dtype=torch.float32)

    b_hidden = numpy.genfromtxt(os.path.join(directory, "b_hidden.csv"), delimiter = ",")
    b_hidden = torch.tensor(b_hidden, dtype=torch.float32)

    self.hidden.set_matrices_and_vias(w_hidden, b_hidden)

The `second neural network` should have the same layers in the same order as the layers used to train the CNN. For example, if we apply the `convolutional layer` multiple times (as mention above), we should write the following:

```
def second_neural_network(self, vector_representation_params):
    # Parameters of the first neural network
    ls_nodes, w_l_code, w_r_code, b_code = vector_representation_params
    # Convolutional layer
    ls_nodes = self.conv1.convolutional_layer(ls_nodes)
    ls_nodes = self.conv2.convolutional_layer(ls_nodes)
    # Pooling layer
    vector = self.pooling.pooling_layer(ls_nodes)
    # Hidden layer
    output = self.hidden.hidden_layer(vector)

    return output
```