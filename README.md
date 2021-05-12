# Tree Based Convolutional Neural Network


## What does the program do?

This program is meant to receive different Python files, and train a Convolutional Neural Network (CNN) to detect patterns 

*so far, it can only dettect generators, but this will be updated in the future to detect other patterns*.

Once we have the trained CNN, we will get as output a folder called `params` that contains each of the matrices and vectors that are necessary to detect patterns in code. 

## Prerequisites

The Python version as well as well as the dependencies are the following:

```
  - python=3.8
  - scipy=1.6.2
  - pytest=6.2.3
  - pandas=1.2.4
  - matplotlib=3.3.4
  - pip 
  - pip:
    - numpy==1.20.0
    - gensim==4.0.1
    - torch==1.8.1
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

**Note**: if you want to use the `main_tester.py` file instead of `main.py` just to test the program instead of training the whole network, the `sets` folder should instead be called `sets_short`.

## How to use the program

Once you have all the preriquisites, you should simply call:

```
python main.py
```

And it will train your neural network based on the data you have in the `sets` folder, or you may also run `main_tester.py` and train the network based on the data in `sets_short`.

Once the program has finished, the `params` folder (initially empty) will contain all the matrices and vectors (in .csv files) with the trained parameters.

*In the future, we will have a program that is capable of receiving this folder and be able to recognize patterns, i.e, we should have a trained neural network using these parameters*.


## How is the program organize?

The program has three independent networks that we call: `first neural network`, `second neural network` and `validation neural network`.

The `first neural network` does the vector representation for all files. Once the neural network is trained, we will have as output a dictionary with the following information: one vector for each node (we will represent the features of each node in a vector), two weighted matrices and one vector bias. It will train the first neural network based on the data you have in the `sets` folder.

*In the furure, we will save all the outputs into a file or similar.*

The `second neural network` received the output of the first neural network as input. The neural network splits the files into two sets: `training set` and `validation set`. The training set has the 70% of the files and is used to train the Convolutional Neural Network (CNN). The output of this neural network is the folder `params` that contains each of the matrices and vectors that are necessary to detect patterns in code. 

*In the furure, we will be able to run each network without running the others networks.*

The way to run each network is using `main.py` or `main_tester.py`. Both scripts contains a function named `main` where the three neural networks are called. To run just one of the neural networks you just have to write its name in the `main` function and run:

```
python main.py
```


## Looking for parameters

To train the neural network, we need to try select different parameters. Since we don't know which parameters work better, we need to try them to see which of them provide lower loss function. 

The way to do this is using `param_tester.py`. The script need to be edited with the corresponding parameters we want to try, and it will call the `main` function with each of these parameters in each iteration, and it will save the results in a file called `results.txt`.

We simply need to run:

```
python param_tester.py
```


**Note**: every time we run this program, it will check if `results.txt` exists (in case we have run the program before and it's saved in our directory). Bear in mind that we will lose this information, since we override the previous results when we call `param_tester.py` again.