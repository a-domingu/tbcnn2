import ast
from node import Node
import random


def node_creator(path):
    module = ast.parse(open(path).read())
    asserter(module)
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


def asserter(path):
    try:
        assert path.__class__.__name__ == 'Module'
    except AssertionError:
        print(path.__class__.__name__)
        print(path)
        raise AssertionError

###############################################
#borar al final


main_node = node_creator('sets\\generators\\withgen\\auth.py')

print(main_node)

print(main_node.children)

for child in main_node.children:
    print('###############33')
    print(child)
    print(child.children)
    print(child.parent)

print('AAAAAAAAAAAAAAAAAAAAAAAAAAAAA')
print('Y ahora vienen los descendientes:')

for descendant in main_node.descendants():
    print(descendant)
