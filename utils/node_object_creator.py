import ast
import sys

from first_neural_network.node import Node


# We create the AST
def file_parser(path):
    return ast.parse(open(path, encoding='utf-8').read())

def node_object_creator(path):
    module = ast.parse(open(path, encoding='utf-8').read())
    module_asserter(module)
    depth = 1
    main_node = Node(module, depth)
    depth+=1
    node_creator_recursive(module, depth, main_node)
    return main_node



def node_creator_recursive(parent_ast, depth, parent_node):
    for child in ast.iter_child_nodes(parent_ast):
        node = Node(child, depth, parent_node)
        parent_node.set_children(node)
        depth+=1
        node_creator_recursive(child, depth, node)


def module_asserter(path):
    try:
        assert path.__class__.__name__ == 'Module'
    except AssertionError:
        print(path.__class__.__name__)
        print(path)
        raise AssertionError
