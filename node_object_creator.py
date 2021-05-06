import ast
import sys

from node import Node


# We create the AST
def file_parser(path):
    return ast.parse(open(path, encoding='utf-8').read())

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

def nodes_vector_update(ls_nodes, w, b):
    for node in ls_nodes:
        node.update_vector(w, b)

# We assign the position (1,..,N) for all the nodes that are at the same hierarchical level (or depth) (under the same parent)
def node_position_assign(ls_nodes):
    for index in range(len(ls_nodes)):
        count  = 1
        good_depth = ls_nodes[index].depth
        # parent = index.parent
        for node in ls_nodes[0:index]:
            # if node.depth == good_depth && parent = node.parent:
            if node.depth == good_depth:
                count+=1
        ls_nodes[index].set_position(count)
        # print("Para el nodo", index, "tenemos depth", ls_nodes[index].depth, "y position", ls_nodes[index].position)
    return ls_nodes


# We assign the number of nodes on the same hierarchical level (under the same parent node), including p itself
def node_sibling_assign(ls_nodes):
    # We create a dictionary where each key is a hierarchical level (depth) in the AST and its items are the number of nodes at each hierarchical level (depth)
    dict_sibling = {}
    for node in ls_nodes:
        if node.depth in dict_sibling.keys():
            dict_sibling[node.depth].append(node) # append(node)
        else:
            dict_sibling[node.depth] = [node]

    # We assing the sibling to each node
    for node in ls_nodes:
        node.set_sibling(dict_sibling[node.depth])
        # print("deph:", node.depth, "position:", node.position, "siblings:", node.siblings)
    
    return ls_nodes, dict_sibling


# We assign the number of leaves nodes under each node
def leaves_nodes_assign(ls_nodes, dict_ast_to_Node):
    for node in ls_nodes:
       leaves_nodes = get_l(node, dict_ast_to_Node)
       node.set_l(leaves_nodes) 
    return ls_nodes


# Calculate the number of leaves nodes under each node
def get_l(node, dict_ast_to_Node):
    '''
    This function's output is the number of leaf nodes under each node
    '''
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