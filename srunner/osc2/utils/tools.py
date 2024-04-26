import re
from typing import Tuple

from antlr4.tree.Tree import TerminalNodeImpl


# node：Input parameter, node of the tree
# nodes：Enter the parameter, number the nodes in traversal order, and line the nodes with the number
# pindex：Id of the parent node
# g：The graph object of graphviz
def render_symbol(node, nodes, pindex, g):
    if not isinstance(node, Tuple):
        name = str(node)

        index = len(nodes)
        nodes.append(index)

        g.node(str(index), name)
        if index != pindex:
            g.edge(
                str(pindex), str(index)
            )  # The edge is from father to son, if there is no upside down tree
        for i in range(0, node.get_number_of_symbols()):
            render_symbol(node.get_child_symbol(i), nodes, index, g)


# The parse tree is printed as LOG
def print_parse_tree(tree, ruleNames, indent=0):
    if isinstance(tree, TerminalNodeImpl):
        if not re.match(r"\r?\n[ \t]*", tree.getText()):
            print("{0}TOKEN '{1}'".format("  " * indent, tree.getText()))
    else:
        print("{0}{1}".format("  " * indent, ruleNames[tree.getRuleIndex()]))
        for child in tree.children:
            print_parse_tree(child, ruleNames, indent + 1)


def multi_field_name_append(names, field_name):
    if names == "":
        names = field_name
    else:
        names = names + "%" + field_name
    return names


def multi_field_name_split(names):
    return names.split("%")


def is_multi_name(names):
    if not names:
        return False
    elif "%" in names:
        return True
    else:
        return False
