import ast
import sys

from node import Node


# We create the AST
def file_parser(path):
    return ast.parse(open(path, encoding='utf-8').read())

def node_object_creator(path):
    module = ast.parse(open(path).read())
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

'''

# We create a list with all AST nodes
def node_object_creator(module):
    dict_ast_to_Node = {} # A dict that relates class ast objects to class Node objects.
    # We assign its hierarchical level (or depth) to the first node (depth = 1)
    depth = 1
    module_node = Node(module, depth)
    if not module in dict_ast_to_Node.keys():
        dict_ast_to_Node[module] = module_node

    ls_nodes = [module_node]
    for child in ast.iter_child_nodes(module):
        ls_nodes = node_object_creator_recursive(module, child, ls_nodes, dict_ast_to_Node, depth)
    return ls_nodes, dict_ast_to_Node

# We instanciate each node as a Node class
def node_object_creator_recursive(parent, node, ls_nodes, dict_ast_to_Node, depth):
    # We assign the hierarchical level (or depth) to each node in the AST
    depth += 1
    new_node = Node(node, depth, parent)
    if not node in dict_ast_to_Node.keys():
        dict_ast_to_Node[node] = new_node

    ls_nodes.append(new_node)
    for child in ast.iter_child_nodes(node):
        ls_nodes = node_object_creator_recursive(new_node, child, ls_nodes, dict_ast_to_Node, depth)
    return ls_nodes

'''



'''

# We assign the number of leaves nodes under each node
def leaves_nodes_assign(ls_nodes, dict_ast_to_Node):
    for node in ls_nodes:
       leaves_nodes = get_l(node, dict_ast_to_Node)
       node.set_l(leaves_nodes) 
    return ls_nodes


# Calculate the number of leaves nodes under each node
def get_l(node, dict_ast_to_Node):
    leaves_under_node = 0
    if len(node.children) == 0:
        return leaves_under_node
    else:
        leaves_nodes = calculate_l(node, leaves_under_node, dict_ast_to_Node)
    return leaves_nodes


def calculate_l(node, leaves_under_node, dict_ast_to_Node):
    #node is a Node object
    #child is an AST object
    for child in node.children:
        child_node = dict_ast_to_Node[child]
        if len(child_node.children) == 0:
            leaves_under_node += 1
        else:
            leaves_under_node = calculate_l(child_node, leaves_under_node, dict_ast_to_Node)
    return leaves_under_node


'''