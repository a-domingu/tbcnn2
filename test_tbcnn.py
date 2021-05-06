import pytest
import numpy as np
import ast
import torch
from gensim.models import Word2Vec

from embeddings import Embedding
from main import training_and_validation_sets_creation, target_tensor_set_up, first_neural_network
from node import Node
from matrix_generator import MatrixGenerator
from node_object_creator import *
from first_neural_network import First_neural_network
from coding_layer import Coding_layer
from convolutional_layer import Convolutional_layer
from pooling_layer import Pooling_layer
from dynamic_pooling import Max_pooling_layer, Dynamic_pooling_layer
from hidden_layer import Hidden_layer
from get_targets import GetTargets
from second_neural_network import SecondNeuralNetwork
from validation_neural_network import Validation_neural_network


@pytest.fixture
def setup_get_targets():
    get_targets = GetTargets('test\labels')
    targets = get_targets.df_iterator()
    return targets
'''
@pytest.fixture
def setup_training_dict():
    training_dict = training_dict_set_up('test')
    return training_dict
'''
@pytest.fixture
def setup_training_validation_sets_creation():
    path = '..\\sets\\generators'
    training_dict, validation_dict = training_and_validation_sets_creation(path) 
    return training_dict, validation_dict

@pytest.fixture
def setup_targets_tensor():
    training_dict, validation_dict = training_and_validation_sets_creation('test')
    targets = target_tensor_set_up('test', training_dict)
    return targets

@pytest.fixture
def setup_first_neural_network():
    training_dict, validation_dict = training_and_validation_sets_creation('test')
    training_dict = first_neural_network(training_dict, 20, 0.1, 0.001)
    return training_dict

@pytest.fixture
def set_up_dictionary():
    tree = file_parser('test\pruebas.py')
    ls_nodes, dict_ast_to_Node = node_object_creator(tree)
    return tree, dict_ast_to_Node

@pytest.fixture
def set_up_embeddings():
    tree = file_parser('test\pruebas.py')
    ls_nodes, dict_ast_to_Node = node_object_creator(tree)
    embed = Embedding(20, ls_nodes, dict_ast_to_Node)
    return embed

@pytest.fixture
def set_up_matrix():
    tree = file_parser('test\pruebas.py')
    ls_nodes, dict_ast_to_Node = node_object_creator(tree)
    embed = Embedding(20, ls_nodes, dict_ast_to_Node)
    ls_nodes = embed.node_embedding()[:]
    matrices = MatrixGenerator(20, 10)
    return matrices

'''
@pytest.fixture
def set_up_update_vector():
    tree = path_to_module('test\pruebas.py')
    ls_nodes, dict_ast_to_Node = node_object_creator(tree)
    embed = Embedding(20, ls_nodes, dict_ast_to_Node)
    ls_nodes = embed.node_embedding()[:]
    matrices = MatrixGenerator(20, 10)
    w, b = matrices.w, matrices.b
    nodes_vector_update(ls_nodes, w, b)
    w, b = matrices.w, matrices.b
    nodes_vector_update(ls_nodes, w, b)
    return ls_nodes
'''

@pytest.fixture
def set_up_vector_representation():
    tree = file_parser('test\pruebas.py')
    ls_nodes, dict_ast_to_Node = node_object_creator(tree)
    embed = Embedding(20, ls_nodes, dict_ast_to_Node)
    ls_nodes = embed.node_embedding()[:]
    vector_representation = First_neural_network(ls_nodes, dict_ast_to_Node, 20, 0.1, 0.001)
    ls_nodes, w_l, w_r, b_code = vector_representation.vector_representation()
    return ls_nodes, w_l, w_r, b_code

@pytest.fixture
def set_up_coding_layer():
    tree = file_parser('test\pruebas.py')
    ls_nodes, dict_ast_to_Node = node_object_creator(tree)
    embed = Embedding(20, ls_nodes, dict_ast_to_Node)
    ls_nodes = embed.node_embedding()[:]
    vector_representation = First_neural_network(ls_nodes, dict_ast_to_Node, 20, 0.1, 0.001)
    ls_nodes, w_l, w_r, b_code = vector_representation.vector_representation()
    w_comb1 = torch.diag(torch.randn(20, dtype=torch.float32)).requires_grad_()
    w_comb2 = torch.diag(torch.randn(20, dtype=torch.float32)).requires_grad_()
    coding_layer = Coding_layer(20, w_comb1, w_comb2)
    ls_nodes = coding_layer.coding_layer(ls_nodes, dict_ast_to_Node, w_l, w_r, b_code)
    return ls_nodes, w_comb1, w_comb2

@pytest.fixture
def set_up_convolutional_layer():
    tree = file_parser('test\pruebas.py')
    ls_nodes, dict_ast_to_Node = node_object_creator(tree)
    ls_nodes = node_position_assign(ls_nodes)
    ls_nodes, dict_sibling = node_sibling_assign(ls_nodes)
    embed = Embedding(20, ls_nodes, dict_ast_to_Node)
    ls_nodes = embed.node_embedding()[:]
    vector_representation = First_neural_network(ls_nodes, dict_ast_to_Node, 20, 0.1, 0.001)
    ls_nodes, w_l, w_r, b_code = vector_representation.vector_representation()
    w_comb1 = torch.diag(torch.randn(20, dtype=torch.float32)).requires_grad_()
    w_comb2 = torch.diag(torch.randn(20, dtype=torch.float32)).requires_grad_()
    coding_layer = Coding_layer(20, w_comb1, w_comb2)
    ls_nodes = coding_layer.coding_layer(ls_nodes, dict_ast_to_Node, w_l, w_r, b_code)
    w_t = torch.randn(4, 20, requires_grad = True)
    w_r = torch.randn(4, 20, requires_grad = True)
    w_l = torch.randn(4, 20, requires_grad = True)
    b_conv = torch.randn(4, requires_grad = True)
    convolutional_layer = Convolutional_layer(20, w_t, w_r, w_l, b_conv, features_size=4)
    ls_nodes = convolutional_layer.convolutional_layer(ls_nodes, dict_ast_to_Node)

    return ls_nodes, w_t, w_l, w_r, b_conv

@pytest.fixture
def set_up_one_max_pooling_layer():
    tree = file_parser('test\pruebas.py')
    ls_nodes, dict_ast_to_Node = node_object_creator(tree)
    ls_nodes = node_position_assign(ls_nodes)
    ls_nodes, dict_sibling = node_sibling_assign(ls_nodes)
    embed = Embedding(20, ls_nodes, dict_ast_to_Node)
    ls_nodes = embed.node_embedding()[:]
    vector_representation = First_neural_network(ls_nodes, dict_ast_to_Node, 20, 0.1, 0.001)
    ls_nodes, w_l, w_r, b_code = vector_representation.vector_representation()
    w_comb1 = torch.diag(torch.randn(20, dtype=torch.float32)).requires_grad_()
    w_comb2 = torch.diag(torch.randn(20, dtype=torch.float32)).requires_grad_()
    coding_layer = Coding_layer(20, w_comb1, w_comb2)
    ls_nodes = coding_layer.coding_layer(ls_nodes, dict_ast_to_Node, w_l, w_r, b_code)
    w_t = torch.randn(4, 20, requires_grad = True)
    w_r = torch.randn(4, 20, requires_grad = True)
    w_l = torch.randn(4, 20, requires_grad = True)
    b_conv = torch.randn(4, requires_grad = True)
    convolutional_layer = Convolutional_layer(20, w_t, w_r, w_l, b_conv, features_size=4)
    ls_nodes = convolutional_layer.convolutional_layer(ls_nodes, dict_ast_to_Node)
    pooling_layer = Pooling_layer()
    pooled_tensor = pooling_layer.pooling_layer(ls_nodes)

    return pooled_tensor

@pytest.fixture
def set_up_dynamic_pooling_layer():
    tree = file_parser('test\pruebas.py')
    ls_nodes, dict_ast_to_Node = node_object_creator(tree)
    ls_nodes = node_position_assign(ls_nodes)
    ls_nodes, dict_sibling = node_sibling_assign(ls_nodes)
    embed = Embedding(20, ls_nodes, dict_ast_to_Node)
    ls_nodes = embed.node_embedding()[:]
    vector_representation = First_neural_network(ls_nodes, dict_ast_to_Node, 20, 0.1, 0.001)
    ls_nodes, w_l, w_r, b_code = vector_representation.vector_representation()
    w_comb1 = torch.diag(torch.randn(20, dtype=torch.float32)).requires_grad_()
    w_comb2 = torch.diag(torch.randn(20, dtype=torch.float32)).requires_grad_()
    coding_layer = Coding_layer(20, w_comb1, w_comb2)
    ls_nodes = coding_layer.coding_layer(ls_nodes, dict_ast_to_Node, w_l, w_r, b_code)
    w_t = torch.randn(4, 20, requires_grad = True)
    w_r = torch.randn(4, 20, requires_grad = True)
    w_l = torch.randn(4, 20, requires_grad = True)
    b_conv = torch.randn(4, requires_grad = True)
    convolutional_layer = Convolutional_layer(20, w_t, w_r, w_l, b_conv, features_size=4)
    ls_nodes = convolutional_layer.convolutional_layer(ls_nodes, dict_ast_to_Node)
    max_pooling_layer = Max_pooling_layer()
    max_pooling_layer.max_pooling(ls_nodes)
    dynamic_pooling = Dynamic_pooling_layer()
    hidden_input = dynamic_pooling.three_way_pooling(ls_nodes, dict_sibling)

    return ls_nodes, hidden_input

@pytest.fixture
def set_up_hidden_layer():
    tree = file_parser('test\pruebas.py')
    ls_nodes, dict_ast_to_Node = node_object_creator(tree)
    ls_nodes = node_position_assign(ls_nodes)
    ls_nodes, dict_sibling = node_sibling_assign(ls_nodes)
    embed = Embedding(20, ls_nodes, dict_ast_to_Node)
    ls_nodes = embed.node_embedding()[:]
    vector_representation = First_neural_network(ls_nodes, dict_ast_to_Node, 20, 0.1, 0.001)
    ls_nodes, w_l, w_r, b_code = vector_representation.vector_representation()
    w_comb1 = torch.diag(torch.randn(20, dtype=torch.float32)).requires_grad_()
    w_comb2 = torch.diag(torch.randn(20, dtype=torch.float32)).requires_grad_()
    coding_layer = Coding_layer(20, w_comb1, w_comb2)
    ls_nodes = coding_layer.coding_layer(ls_nodes, dict_ast_to_Node, w_l, w_r, b_code)
    w_t = torch.randn(4, 20, requires_grad = True)
    w_r = torch.randn(4, 20, requires_grad = True)
    w_l = torch.randn(4, 20, requires_grad = True)
    b_conv = torch.randn(4, requires_grad = True)
    convolutional_layer = Convolutional_layer(20, w_t, w_r, w_l, b_conv, features_size=4)
    ls_nodes = convolutional_layer.convolutional_layer(ls_nodes, dict_ast_to_Node)
    max_pooling_layer = Max_pooling_layer()
    max_pooling_layer.max_pooling(ls_nodes)
    dynamic_pooling = Dynamic_pooling_layer()
    hidden_input = dynamic_pooling.three_way_pooling(ls_nodes, dict_sibling)
    w_hidden = torch.randn(3, requires_grad = True)
    b_hidden = torch.randn(1, requires_grad = True)
    hidden = Hidden_layer(w_hidden, b_hidden)
    output_hidden  = hidden.hidden_layer(hidden_input)

    return output_hidden, w_hidden, b_hidden


@pytest.fixture
def setup_second_neural_network():
    training_dict, validation_dict = training_and_validation_sets_creation('test')
    targets = target_tensor_set_up('test', training_dict)
    training_dict = first_neural_network(training_dict, 20)
    secnn = SecondNeuralNetwork(20, 4)
    outputs = secnn.forward(training_dict)
    return outputs


@pytest.fixture
def setup_validation_neural_network():
    training_dict, validation_dict = training_and_validation_sets_creation('test')
    targets = target_tensor_set_up('test', training_dict)
    training_dict = first_neural_network(training_dict, 20)
    secnn = SecondNeuralNetwork(20, 4)
    secnn.train(targets, training_dict)
    val = Validation_neural_network(20, 4)
    targets = val.target_tensor_set_up('test', validation_dict)
    predicts = val.prediction(validation_dict)
    accuracy = val.accuracy(predicts, targets)
    return predicts, accuracy


def test_get_targets(setup_get_targets):
    targets = setup_get_targets
    assert isinstance(targets, dict)
    assert targets != {}
    for target_key in targets:
        target = targets[target_key]
        break
    assert isinstance(target, torch.Tensor)
    assert len(target.shape) == 1
    assert target.shape[0] == 1
    assert target.numpy()[0] == 0
    
'''
def test_training_dict(setup_training_dict):
    training_dict = setup_training_dict
    assert isinstance(training_dict, dict)
    assert training_dict != {}
'''

def test_training_validation_sets_creation(setup_training_validation_sets_creation):
    training_dict, validation_dict = setup_training_validation_sets_creation
    assert isinstance(training_dict, dict)
    assert isinstance(validation_dict, dict)
    assert training_dict != {}
    assert validation_dict != {}


def targets_tensor_dict(setup_targets_tensor):
    targets = setup_targets_tensor
    assert isinstance(targets, torch.Tensor)
    assert len(targets) == 1


def test_first_neural_network(setup_first_neural_network):
    training_dict = setup_first_neural_network
    assert isinstance(training_dict, dict)
    assert training_dict != {}
    for data in training_dict:
        data = training_dict[data]
        break
    assert isinstance(data, list)
    assert len(data) == 6
    assert isinstance(data[0], list)
    assert isinstance(data[1], dict)
    assert isinstance(data[2], dict)
    assert isinstance(data[3], torch.Tensor)
    assert isinstance(data[4], torch.Tensor)
    assert isinstance(data[5], torch.Tensor)


def test_dictionary_Node(set_up_dictionary):

    tree, dict_ast_to_Node = set_up_dictionary

    for node in ast.iter_child_nodes(tree):
        assert node in dict_ast_to_Node
        assert dict_ast_to_Node[node].__class__.__name__ == "Node"

# Error a solucionar (file Embeddings.py line 37)
# Hay un error porque no reconoce como argumento "vector_size" en el comando word2vec. 
# Al escribir "size" como argumento funcionan los test pero da error al ejecutar el código. 
# Viceversa cuando escribimos "vector_size" funciona el código pero da error en los tests.
def test_node_embedding(set_up_embeddings):

    result = set_up_embeddings.node_embedding()[:]
    length_expected = 20

    for el in result:
        assert len(el.vector) == length_expected
                        
def test_matrix_length(set_up_matrix):
    
    w, b = set_up_matrix.w, set_up_matrix.b

    assert w.shape == (20, 10)
    assert len(b) == 20

'''
def test_update_vector(set_up_update_vector):
    for node in set_up_update_vector:
        assert len(node.new_vector) > 0
'''
def test_vector_representation(set_up_vector_representation):
    
    ls_nodes, w_l, w_r, b_code = set_up_vector_representation
    feature_size_expected = 20

    for node in ls_nodes:
        vector = node.vector.detach().numpy()
        assert len(vector) == feature_size_expected
        assert np.count_nonzero(vector) != 0
    
    assert w_l.shape == (feature_size_expected, feature_size_expected)
    w_l = w_l.detach().numpy()
    assert np.count_nonzero(w_l) != 0
    assert w_r.shape == (feature_size_expected, feature_size_expected)
    w_r = w_r.detach().numpy()
    assert np.count_nonzero(w_r) != 0
    assert len(b_code) == feature_size_expected

def test_coding_layer(set_up_coding_layer):
    
    ls_nodes, w_comb1, w_comb2 = set_up_coding_layer
    feature_size_expected = 20

    for node in ls_nodes:
        assert len(node.combined_vector) == feature_size_expected
        vector = node.combined_vector.detach().numpy()
        assert np.count_nonzero(vector) != 0
    
    assert w_comb1.shape == (feature_size_expected, feature_size_expected)
    w_comb1 = w_comb1.detach().numpy()
    assert np.count_nonzero(w_comb1) != 0
    assert w_comb2.shape == (feature_size_expected, feature_size_expected)
    w_comb2 = w_comb2.detach().numpy()
    assert np.count_nonzero(w_comb2) != 0


def test_convolutional_layer(set_up_convolutional_layer):
    
    ls_nodes, w_t, w_l, w_r, b_conv = set_up_convolutional_layer
    feature_size_expected = 20
    output_size_expected = 4

    for node in ls_nodes:
        assert len(node.y) == output_size_expected
    
    assert w_t.shape == (output_size_expected, feature_size_expected)
    w_t = w_t.detach().numpy()
    assert np.count_nonzero(w_t) != 0
    assert w_l.shape == (output_size_expected, feature_size_expected)
    w_l = w_l.detach().numpy()
    assert np.count_nonzero(w_l) != 0
    assert w_r.shape == (output_size_expected, feature_size_expected)
    w_r = w_r.detach().numpy()
    assert np.count_nonzero(w_r) != 0
    assert  len(b_conv) == output_size_expected


def test_one_max_pooling_layer(set_up_one_max_pooling_layer):
    pooled_tensor = set_up_one_max_pooling_layer
    expected_dimension = 1
    expected_size = 4
    assert isinstance(pooled_tensor, torch.Tensor)
    assert len(pooled_tensor.shape) == expected_dimension
    assert pooled_tensor.shape[0] == expected_size


def test_dynamic_pooling_layer(set_up_dynamic_pooling_layer):
    
    ls_nodes, hidden_input = set_up_dynamic_pooling_layer
    for node in ls_nodes:
        pool = node.pool.detach().numpy()
        assert pool.size == 1
    assert len(hidden_input) == 3


def test_hidden_layer(set_up_hidden_layer):
    
    output_hidden, w_hidden, b_hidden = set_up_hidden_layer

    assert len(output_hidden) == 1
    output_hidden = output_hidden.detach().numpy()
    assert np.count_nonzero(output_hidden) != 0
    assert len(w_hidden) == 3
    w_hidden = w_hidden.detach().numpy()
    assert np.count_nonzero(w_hidden) != 0
    assert  len(b_hidden) == 1


def test_second_neural_network(setup_second_neural_network):
    
    outputs = setup_second_neural_network

    assert isinstance(outputs, torch.Tensor)
    assert len(outputs) == 2
    assert 0 <= outputs[0] <= 1


def test_validation(setup_validation_neural_network):
    
    predicts, accuracy = setup_validation_neural_network

    assert isinstance(predicts, torch.Tensor)
    assert len(predicts) == 1
    assert 0 <= predicts <= 1
    assert isinstance(accuracy, torch.Tensor)
    assert 0 <= accuracy <= 1