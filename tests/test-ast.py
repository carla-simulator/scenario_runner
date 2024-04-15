import os
import sys
from os.path import abspath, dirname
from typing import Tuple
from antlr4 import *
sys.path.append(os.getcwd())
try:
    sys.path.append('../')
except IndexError:
    pass
from srunner.osc2.osc_preprocess.pre_process import ImportFile, Preprocess
from srunner.osc2.ast_manager.ast_node import AST
import srunner.osc2.ast_manager.ast_node as ast_node

from srunner.osc2.utils.log_manager import *
from srunner.osc2.utils.tools import *
from srunner.osc2.error_manager.error_listener import OscErrorListener
from srunner.osc2.osc2_parser.OpenSCENARIO2Lexer import OpenSCENARIO2Lexer
from srunner.osc2.osc2_parser.OpenSCENARIO2Parser import OpenSCENARIO2Parser
from srunner.osc2.ast_manager.ast_builder import ASTBuilder

import graphviz

# node：node of the tree
# nodes：number the nodes in traversal order and line them with numbers
# pindex：id of the parent node
# g：graphviz
def render_ast(node, nodes, pindex, g):
    if not isinstance(node, Tuple):
        name = str(node)

        index = len(nodes)
        nodes.append(index)

        g.node(str(index), name)
        if index != pindex:
            # g.edge(str(index), str(pindex))
            g.edge(str(pindex), str(index))
        if isinstance(node, ast_node.AST):
            for i in range(0, node.get_child_count()):
                render_ast(node.get_child(i), nodes, index, g)
 
def main(input_stream, output_name):
    quiet = False
    OscErrorListeners = OscErrorListener(input_stream)
    lexer = OpenSCENARIO2Lexer(input_stream)
    lexer.removeErrorListeners()
    lexer.addErrorListener(OscErrorListeners)
    stream = CommonTokenStream(lexer)

    parser = OpenSCENARIO2Parser(stream)
    parser.removeErrorListeners()
    parser.addErrorListener(OscErrorListeners)
    tree = parser.osc_file()
    errors = parser.getNumberOfSyntaxErrors()

    if errors == 0:
        ast_listener = ASTBuilder()
        walker = ParseTreeWalker()
        walker.walk(ast_listener, tree)

        graph = graphviz.Graph(node_attr={'shape': 'plaintext'}, format='png')
        render_ast(ast_listener.ast, [], 0, graph)
        # graph.view() #generate and view graph
        #graph.render(cleanup=True, outfile=output_name.split('\\')[-1]+".png") #generate and save graph, but not view

    if not quiet:
        LOG_INFO("Errors: "+ str(errors))
        #print_parse_tree(tree, parser.ruleNames)
    return errors


if __name__ == '__main__':
    error_file_list = []
    if not os.path.exists(sys.argv[1]):
        print("File path error")
    if os.path.isdir(sys.argv[1]):
        filepath = sys.argv[1]
        files = os.listdir(filepath)
        files.sort()
        for fi in files:
            fpath = os.path.join(filepath, fi)
            LOG_INFO("======================== "+fi+" ========================")
            new_file, import_msg = Preprocess(fpath).import_process()
            input_stream = FileStream(new_file, encoding='utf-8')
            if main(input_stream, fpath)>0:
                error_file_list.append(fi)
            import_msg.clear_msg()

        LOG_INFO("======================== "+"error file result"+" ========================")
        for error_file in error_file_list:
             LOG_INFO(error_file)

    elif os.path.isfile(sys.argv[1]):
        new_file, import_msg = Preprocess(sys.argv[1]).import_process()
        input_stream = FileStream(new_file, encoding='utf-8')
        if main(input_stream, sys.argv[1]) > 0:
            LOG_INFO("======================== "+"error file result"+" ========================")
            LOG_INFO(sys.argv[1])
            pass
    else:
        pass

