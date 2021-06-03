# Tree Based Convolutional Neural Network


## What does the program do?

This program is meant to receive different Python files, and train a Convolutional Neural Network (CNN) to detect patterns 

*so far, it can only dettect generators, but our intention is to be able to detect more patterns (wrapper, decorator, observer...)*.

Once we have the trained CNN, we will get as output a folder called `params` that contains each of the matrices and vectors that are necessary to detect patterns in code. 

## Prerequisites

The Python version as well as well as the dependencies are the following:

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

You also need to have the following in your directory: a folder called `sets` which should contain a `generators` subfolder, which should contain all the data you're going to use to train the neural network. It should only contain Python files, and you should divide them on whether they have generators or not by placing them in the `withgen` or `nogen` accordingly. 

To summarize, the structure should be the following:

```
sets.
└───generators
    ├───nogen
    └───withgen
```

**Note**: if you want to use the `main_tester.py` file instead of `main.py` just to test the program instead of training the whole network, the `sets` folder should instead be called `sets_short`, or you may also edit the path to the generators manually within the file.

## How to use the program

Once you have all the preriquisites, you should simply call:

```
python main.py
```

And it will train your neural network based on the data you have in the `sets` folder, or you may also run `main_tester.py` and train the network based on the data in `sets_short`.

Once the program has finished, the `params` folder will contain all the matrices and vectors (in .csv files) with the trained parameters.

## How is the program organized?

The program has two independent networks that we call: `first neural network` and `second neural network`

The `first neural network` does the vector representation for all files. We should note that because our program read ASTs, and not vectors, in order to use a CNN, we first need to convert each node into a vector. We make this using another neural network (the first one), which takes the idea behind `Word2Vec`, i.e, creates the vectors based on the overall structure of the tree. Once the neural network is trained, we will have as output a dictionary with the following information: one vector for each node (we will represent the features of each node in a vector), two weighted matrices and one vector bias. It will train the first neural network based on the data you have in the `sets` folder. All these parameters are saved for each file using `pickle` into the `vector_representation` folder. 

Before using this neural network, we need a first vector representation done with `Word2Vec`. In order to do this, simply call `python initialize_vector_representation.py` and it will generate a file called `initial_vector_representation.py` with the vector for each type of node. Then you can call `python main_first_neural_network.py` and it will use this initial vectors to get a better vector representation learning the context of each node within the tree.

Once you have this vectors, we can now go to the second neural network. The `second neural network` receives the output of the first neural network as input. The neural network splits the files into two sets: `training set` and `validation set`. The training set has the 70% of the files and is used to train the Convolutional Neural Network (CNN). The output of this neural network is the folder `params` that contains each of the matrices and vectors that are necessary to detect patterns in code. 

You also need to have the following in your directory: a folder called `confusion_matrix`. After each iteration, the neural network will test the accuracy of its parameters by using the `validation set` and it will record its confusion matrix in the folder called `confusion_matrix`.

The way to run this CNN is by calling `pyhton main_second_neural_network.py`. You need to enter this file and adjust the parameters accordingly, and at the end it will save all the trained matrices into the aforementioned `params` folder. You will also get in your screen the loss of the training set and the validation set, as well as the accuracy (proportion of correctly predicted files), and a small representation of the confusion matrix. You also have a `confusion_matrix` folder with each of the confusion matrices that yielded a better result than the previous epochs.

## Looking for parameters

To train the neural network, we need to try select different parameters. Since we don't know which parameters work better, we need to try them to see which of them provide lower loss function. 

The way to do this is using `param_tester.py`. The script need to be edited with the corresponding parameters we want to try, and it will call the `main` function with each of these parameters in each iteration, and it will save the results in a file called `results.txt`.

We simply need to run:

```
python param_tester.py
```


**Note**: every time we run this program, it will check if `results.txt` exists (in case we have run the program before and it's saved in our directory). Bear in mind that we will lose this information, since we override the previous results when we call `param_tester.py` again.


## How to check if a python file has generators?

Once you have your neural network trained, you should simply call:
```
python generator_detector.py
```

You should specify as input the path of a file (or a folder) you are interested in. It also will use the folder `params` as input to be able to detect generators in code. 

Once the program has finished, it will return as output if there is a generator in a particular file or not.