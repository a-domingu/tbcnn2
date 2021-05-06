import ast
import random
import numpy as np

from gensim.models import Word2Vec

class Embedding():
    '''
    In this class we initialize vec(·), where vec(·) is the feature representation of a node in the AST.
    We use random walks to learn the context information of each node and then we apply word2vec to translate
    it into a vector.
    Inputs:
    walkLength [int]: Maximum number of nodes visited on each random walk
    windowSize [int]: Parameter of word2vec model
    size [int]: Vector embedding size
    minCount [int]: Paramenter of word2vec model
    ls_nodes [list <class Node>]: list with all nodes in the AST
    dict_ast_to_Node[dict[ast_object] = <class Node>]: dictionary that relates class ast objects to class Node objects
    Output:
    ls_nodes [list <class Node>]: We assign a vector embedding to each node
    '''

    def __init__(self, size, ls_nodes, dict_ast_to_Node, walkLength = 10, windowSize = 5, minCount = 1):
        self.walkLength = walkLength
        self.window = windowSize
        self.size = size
        self.minCount = minCount
        self.ls = ls_nodes
        self.dict_ast_to_Node = dict_ast_to_Node
        # self.embedding = self.node_embedding()

    #We apply word2vec that returns a vector associated to a node type
    def node_embedding(self):
        matrix = self.generateWalkFile()
        model = Word2Vec(matrix, vector_size = self.size, min_count = self.minCount, window = self.window)
        self.saveVectors(model)
        return self.ls

    #We create a list where each element is a random walk
    def generateWalkFile(self):
        walkMatrix = []
        #We create a random walk for each node
        for node in self.ls:
            walk = self.randomWalk(node)
            walkMatrix.append(walk)
        return walkMatrix

    #Random walk 
    def randomWalk(self, node):
        walkList= []
        #We visited randomly one node child
        while(len(walkList) < self.walkLength):
            walkList.append(str(node.type))
            if node.children: 
                #We choose randomly an ast object
                node = random.choice(node.children)
                #We convert the ast object to an Node object
                node = self.dict_ast_to_Node[node]
            else:
                break
        return walkList
        
    #We assign its vector embedding based on the node type to each node   
    def saveVectors(self, model):
        for node in self.ls:
            vector = model.wv[node.type]
            node.set_vector(vector)
            #print(model.wv[node.type])
