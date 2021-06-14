# New patterns

## How to train the neural network to detect new patterns?

To train the neural network you should have a data set and a test set for the new pattern. Also, you should create the following subclasses: a second neural network subclass, a pattern test subclass and a pattern detector subclass for this new pattern.

**Note**: The structure of the data set and test set should follow the structure explained in the file `Instructions.md` and section `Data set`.


## Second neural network abstract class

The `second neural network` is an abstract class that allows us to train the CNN to detect patterns based on the data set. All the second neural network subclasses has multiple common functions. However, each pattern has three particular functions: `matrices and layers initialization`, `layers` and `save`.

In the `matrices and layers initialization` function, you can choose the layers you want to use as well as the number of convolutional layers. Also, the function should return all the matrices and vectors used to detect patterns. The `layers` function should have all the layers you want to use and you should write them in order. The `save` function save all the trained matrices and vectors (in a .csv) in a subfolder of the `params` folder.


## Pattern test abstract class

The `pattern test` is an abstract class that allows us to test the accuracy of the CNN to detect patterns in code. All the pattern test subclasses has multiple common functions. However, each pattern has two particular functions: `load matrices and vectors` and `second neural network`.

The `load matrices and vectors` function reads all the trained matrices and vectors that are necessary to detect patterns in code. The `second neural network` should have the same layers in the same order as the layers used to train the CNN.


## Pattern detector abstract class

The `pattern detector` is an abstract class that allows us to check if a local file (or folder) or a GitHub repository constains the required pattern. All the pattern detecto subclasses has multiple common functions. However, each pattern has two particular functions: `load matrices and vectors` and `second neural network`.

The `load matrices and vectors` function reads all the trained matrices and vectors that are necessary to detect patterns in code. The `second neural network` should have the same layers in the same order as the layers used to train the CNN.