# DISCERN FRAMEWORK PROJECT

## About the project

The Discern Framework Project is an artificial intelligence software project aimed to detect software patterns in source code. It is applied to complete software projects. It is based on a TBCNN (Tree Based Convolutional Neural Network), and it is developed in the European Industrial Center at Capgemini, Spain. The project is developed as an I+D+I project at Capgemini, with the final goal to be included in the DECODER application (REF) as a tool for software development assistance.

The Discern Framework is a project that provides a tool developed with Deep Learning tecniques and Machine Learning algorithms. The framework offers to the user the detection of software design patterns in arbitrary source code. The framework is meant to receive different Python files, and train a Convolutional Neural Network (CNN) to detect patterns. The TBCNN is a specific type of CNN that works on Abstract Syntax Tree (AST) of source code. To be able to train the CNN, we previously transform each node of the AST into a vector.

The TBCNN has two independent networks:

 - The **first neural network** does the embedding and the vector representation for each AST.
 - The **second neural network** apply several transformations to train the Convolutional Neural Network (CNN).

If you are interested on how to use the program, you can find more information in the `Instructions.md` file. You will also find some class diagrams which describes the structure of the system in the `class_diagram` folder. Moreover, if you want to know how to update the web application, please read carefully the `WebApp.md` file.

Currently, the Discern Framework can only dettect generators, but our intention is to be able to detect other patterns. In fact, the framework was built as an extensible software with the aim of detect new patterns in a simple way (*Please, check the file Patterns.md if you are insterested in add a new pattern*).


## The Team

We are a team of three interns finishing our bachelors on Mathematics and Physics, specialising on Machine Learning and Artificial Intelligence at Capgemini. Also, a Senior Software Engineer is working with us.

Hardworking members:

 - Alfonso Dominguez De La Iglesia

   *Machine Learning Scientist (B.A. Mathematics)*

 - David Minet Casado

   *Machine Learning Scientist (B.A. Physicist)*

 - Esther Pla Ibañez

   *Machine Learning Scientist (MSc. on Mathematics)*

 - Ignacio Gallego Sagastume

   *Project Coordinator (Senior Software Engineer)*


## Contact us

Please open an issue or contact tbcnn.datadons@gmail.com with any questions.


Shipping Open Source Software with MIT licence 2.0 by Capgemini Spain