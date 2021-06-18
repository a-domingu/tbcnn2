# Tree Based Convolutional Neural Network


## What does the program do?

This program is meant to receive different Python files, and train a Convolutional Neural Network (CNN) to detect patterns. The training of the CNN is based on the Abstract Syntax Tree (AST) of the source code. To train the CNN, we previously transform each node of the AST into a vector.

Once you trained the CNN to detect a specific pattern, you will get as output a folder named as the pattern (for example `generator`). This folder will contain each of the matrices and vectors that are necessary to detect patterns in code. These matrices and vectors should have a good accurary in the validation to be able to detect patterns in code.

In your directory it will be a folder called `params` that contains the `generator` subfolder. If the CNN is trained to detect a new pattern, a new subfolder named as the pattern will be created in the folder `params`.

To summarize, the structure will be the following:

```
params.
├───generator
├───wrapper
└───...
    
```

Currently, the Discern Framework can only dettect generators, but our intention is to be able to detect other patterns. In fact, the framework was built as an extensible software with the aim of detect new patterns in a simple way (*Please, check the file Patterns.md if you are insterested in add a new pattern*).

## Prerequisites

The Python version as well as the dependencies are the following:

```
  - python=3.8
  - scipy=1.6.2
  - pytest=6.2.3
  - pandas=1.2.4
  - matplotlib=3.3.4
  - git=2.23.0
  - pip 
  - pip:
    - numpy==1.20.0
    - gensim==4.0.1
    - torch==1.8.1
    - GitPython==3.1.17
```

We recommend you install these dependencies using `conda`, and they can be found in `dependencies.yml`. 


## Data set

You also need to have the following in your directory: a folder called `sets` which should contain a `generator` subfolder, which should contain all the data you're going to use to train the neural network. It should only contain Python files, and you should divide them on whether they have generators or not by placing them in the `withpattern` or `nopatter` accordingly. 

To summarize, the structure should be the following:

```
sets.
└───generator
    ├───nopattern
    └───withpattern
```

In our case, we used a data set with 400 files: 200 files for each label. In general, we recommend you to use around 400 files to train each pattern.

**Note**: if you want to train the CNN to detect other patterns, for example wrappers, you should create a `wrapper` subfolder. The folder should contain all the data (only Python files) you're going to use to train the neural network and you should divide it on whether they have wrappers or not by placing them in the `withpattern` or `nopatter` accordingly.

The structure will be the following: 

```
sets.
├───generator
|   ├───nopattern
|   └───withpattern
└───wrapper
    ├───nopattern
    └───withpattern
```


## Parameters

The parameters used to train the CNN are the following:

```
 - folder = sets
 - pattern = generator
 - vector_size = 30
 - learning_rate = 0.3
 - momentum = 0
 - l2_penalty = 0
 - epoch_first = 1
 - learning_rate2 = 0.001
 - feature_size = 100
 - epoch = 5
 - batch = 20
```

The functionality of each parameter will be explained below:

 - Folder: You can specify the name of your data set. We recommend you to called it `sets` but you can use a different name. *IMPORTANT: The folder should have the structure explained in the Dataset section.*
 - Pattern: You should specify which pattern you're going to use to train the CNN.
 - Vector size: You can choose the size of the vector representation.
 - Learning rate: You can choose the learning rate for the vector representation.
 - Momentum: You can choose the parameter (epsilon) used in the SGD with momentum algorithm.
 - l2 penalty: You can choose the hyperparameter that strikes the balance between coding error and l2 penalty.
 - Epoch first: You can choose the number of epochs for the vector representation.
 - Learning rate 2: You can choose the learning rate for the TBCNN.
 - Feature size: You can choose the number of features detectors.
 - Epoch: You can choose the number of epochs for the TBCNN.
 - Batch: You can choose the batch size.


You also have a `parameters.py` file in your directory, which contains the previous parameters. In case you want to train the CNN using other parameters, you can edit the `parameters.py` file and modify the values. *IMPORTANT: You should only modify the value, not the name of the parameter.*

**Note**: If you just want to test the program instead of training the whole network, the folder value should be `sets_short` instead of `sets`.


## How is the program organized?

The program has two independent networks that we call: `first neural network` and `second neural network`

### First neural network

The `first neural network` does the vector representation for all files. We should note that because our program read ASTs, and not vectors, in order to use a CNN, we first need to convert each node into a vector. We make this using another neural network (the first one), which takes the idea behind `Word2Vec`, i.e, creates the vectors based on the overall structure of the tree. Once the neural network is trained, we will have as output a dictionary with the following information: one vector for each node (we will represent the features of each node in a vector), two weighted matrices and one vector bias. It will train the first neural network based on the data you have in the `sets` folder. All these parameters are saved for each file using `pickle` into the `vector_representation` folder. 

Before using this neural network, we need a first vector representation done with `Word2Vec`. In order to do this, simply call `python initialize_vector_representation.py` and it will generate a file called `initial_vector_representation.csv` with the vector for each type of node. Then you can call `python call_vector_representation.py` and it will use this initial vectors to get a better vector representation learning the context of each node within the tree.

### Second neural network

Once you have this vectors, we can now go to the second neural network. The `second neural network` receives the output of the first neural network as input. The neural network splits the files into two sets: `training set` and `validation set`. The training set has the 70% of the files and is used to train the Convolutional Neural Network (CNN). The output of this neural network is the folder `params` that contains each of the matrices and vectors that are necessary to detect patterns in code. 

You also need to have the following in your directory: a folder called `confusion_matrix`. After each iteration, the neural network will test the accuracy of its parameters by using the `validation set` and it will record its confusion matrix in the folder called `confusion_matrix`.

The way to run this CNN is by calling `pyhton pattern_training.py`. This file will read the parameters from the `parameters.txt` file, and at the end it will save all the trained matrices into the aforementioned `params` folder. You will also get in your screen the loss of the training set and the validation set, as well as the accuracy (proportion of correctly predicted files), and a small representation of the confusion matrix. You also have a `confusion_matrix` folder with each of the confusion matrices that yielded a better result than the previous epochs.


## How to use the program

Once you have all the preriquisites and datasets, you can run the program step by step (as mention above). To do that you should simple call:

```
python initialize_vector_representation.py
python call_vector_representation.py
python pattern_training.py
```

Also, you can call just some steps independently. For example, if you just want to train the `second neural network` but keep the vector representation for each file, you can simply call:

```
python pattern_training.py
```

Moreover, there is another script called TBCNN.py that allows you to run all the steps in one time. You simple need to run:
<!---This way allow you to use just some steps independently or even repeat more than one time a single step. The other way is to use another script called TBCNN.py that allows you to run all the steps in one time. You simple need to run:--->

```
python TBCNN.py
```

And it will train your neural network based on the data you have in the `sets` folder. The neural network will be trained using the parameters you have in the `parameters.py` file. Running the `TBCNN.py` file you will get the file `initial_vector_representation.csv`, the vector representation for each node and each file; and all the trained matrices saved into the `params` folder.

Once the program has finished, the `params` folder will contain a subfolder that contains all the matrices and vectors (in .csv files) with the trained parameters.


## Looking for parameters

To train the neural network, we need to try select different parameters. Since we don't know which parameters work better, we need to try them to see which of them provide lower loss function. 

The way to do this is using `param_tester.py`. The script need to be edited with the corresponding parameters we want to try, and it will call the `vector_representation` and `pattern_training` functions with each of these parameters in each iteration, and it will save the results in a file called `results.txt`.

We simply need to run:

```
python param_tester.py
```


**Note**: every time we run this program, it will check if `results.txt` exists (in case we have run the program before and it's saved in our directory). Bear in mind that we will lose this information, since we override the previous results when we call `param_tester.py` again.


## How to test the pattern's accuracy?

You should have the following in your directory: a folder called `test_sets`, which should contain a `pattern` subfolder. Also, this `pattern` subfolder should contain all the data you are going to use to test the accuracy of your neural network. It should only contain Python files, and you should divide them on whether they have the required pattern or not by placing them in the `withpattern` or `nopatter` accordingly.  

To summarize, the structure should be the same as the data set. Example for generators:

```
test_sets
└───generator
    ├───nopattern
    └───withpattern
```

Once you have your test set, you should simply call:

```
python test_accuracy_pattern.py pattern
```

where `pattern` is the pattern that you want to test (for example generator, wrapper, decorator...). The program will use the parameters you have in the `parameters.py` file to do the `vector representation` for each file of the `test set`. Also, the `test_accuracy_pattern.py` file will use the matrices and vectors that are contained in the folder `params` as input, to be able to detect the required pattern in code. 

At the end you will get in your screen the following elements: the accuracy (proportion of correctly predicted files) based on the test set, a small representation of the confusion matrix and you will know which files were not well predicted. 

If your neural network have a good accuracy (up to 90%), then your CNN is well trained. In this case, you can save the matrices and vectors used to detect the pattern, the `initial_vector_representation.csv` with the vector representation for each type of node; and the parameters used to get the vector representation of a given file. These parameters are:

```
 - vector_size 
 - learning_rate 
 - momentum
 - l2_penalty 
 - epoch_first 
```

The matrices and vectors used to detect the pattern are saved into the `params` folder. However, to save the `initial_vector_representation.csv` and the parameters used to get the `vector representation` of a given file, you need the following folder structure:

```
initial_parameters.
├───generator
|   ├───inital_vector_representation.csv
|   └───parameters_first_neural_network.py
└───wrapper
    ├───inital_vector_representation.csv
|   └───parameters_first_neural_network.py
```

In your working directory you will find the `initial_parameters` folder. As you can see above, this folder contains a subfolder for each pattern. In the pattern subfolder (generator, wrapper,...) you will find two files: `initial_vector_representation.csv` and `parameters_first_neural_network.py`. 

The file `parameters_first_neural_network.py` contains the parameters that are used to get the `vector representation` for a given file. In the pattern `generator` case, these parameters are:

```
 - vector_size = 30
 - learning_rate = 0.3
 - momentum = 0
 - l2_penalty = 0
 - epoch_first = 1
```

**Note**: if you train the CNN to detect other patterns, for example decorators, you should create a `decorator` subfolder. The folder should contain the `initial_vector_representation.csv` used to train the CNN and a `parameters_first_neural_network.py` file with the parameters used to get the `vector representation` of each file.


## How to check if a python file has the required pattern?

Once you have your neural network trained for this pattern, you should simply call:

```
python generator_detector.py pattern 
```

where `pattern` is the pattern that you want to detect (for example generator, wrapper, decorator...). The program will ask you by screen the following: Do you want to indicate the path to a local file (or folder) or do you want to indicate a URL (GitHub repository)? . It also will use the folder `params` as input to be able to detect the required pattern in code. 

Once the program has finished, it will return as output which files have the required pattern.


## How to train the neural network to detect new patterns?

*Please check the file `Patterns.md`.*