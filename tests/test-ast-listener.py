import os
import sys
from os.path import abspath, dirname
from typing import Tuple
from antlr4 import *

try:
    sys.path.append('../')
except IndexError:
    pass

from srunner.osc2.ast_manager.ast_node import AST
import srunner.osc2.ast_manager.ast_node as ast_node

from srunner.osc2.utils.log_manager import *
from srunner.osc2.utils.tools import *
from srunner.osc2.error_manager.error_listener import OscErrorListener
from srunner.osc2.osc2_parser.OpenSCENARIO2Lexer import OpenSCENARIO2Lexer
from srunner.osc2.osc2_parser.OpenSCENARIO2Parser import OpenSCENARIO2Parser
from srunner.osc2.ast_manager.ast_builder import ASTBuilder

from srunner.osc2.ast_manager.ast_listener import ASTListener
from srunner.osc2.ast_manager.ast_walker import ASTWalker
from srunner.osc2.osc_preprocess.pre_process import ImportFile, Preprocess

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

class InvocationTest(ASTListener):

    def __init__(self):
        self.arguments = []
        self.__value = None

    def enter_modifier_invocation(self,  node: ast_node.ModifierInvocation):
        #print("enter modifier invocation!")
        # print("modifier name:", node.modifier_name)
        self.arguments = []
        self.__value = None

    def exit_modifier_invocation(self,  node: ast_node.ModifierInvocation):
        #print("exit modifier invocation!")
        #print(self.arguments)
        pass

    def enter_named_argument(self,  node: ast_node.NamedArgument):
        self.__value = None

    def exit_named_argument(self,  node: ast_node.NamedArgument):
        self.arguments.append((node.argument_name, self.__value))

    def enter_physical_literal(self,  node: ast_node.PhysicalLiteral):
        self.arguments.append((node.value, node.unit_name))

    def exit_physical_literal(self,  node: ast_node.PhysicalLiteral):
        pass

    def enter_integer_literal(self,  node: ast_node.IntegerLiteral):
        self.__value = node.value

    def exit_integer_literal(self,  node: ast_node.IntegerLiteral):
        pass

    def enter_float_literal(self,  node: ast_node.FloatLiteral):
        self.__value = node.value

    def exit_float_literal(self,  node: ast_node.FloatLiteral):
        pass

    def enter_bool_literal(self,  node: ast_node.BoolLiteral):
        self.__value = node.value

    def exit_bool_literal(self,  node: ast_node.BoolLiteral):
        pass

    def enter_string_literal(self,  node: ast_node.StringLiteral):
        self.__value = node.value

    def exit_string_literal(self,  node: ast_node.StringLiteral):
        pass

    def enter_identifier(self,  node: ast_node.Identifier):
        self.__value = node.name

    def exit_identifier(self,  node: ast_node.Identifier):
        pass
 
def main(input_stream):
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
        build_ast = ASTBuilder()
        walker = ParseTreeWalker()
        walker.walk(build_ast, tree)
        ast = build_ast.get_ast()
        symbol = build_ast.get_symbol()

        graph = graphviz.Graph(node_attr={'shape': 'plaintext'},format='png')
        #render_symbol(symbol, [], 0, graph)
        #render_ast(ast, [], 0, graph)
        #graph.view()

        invocation_walker = ASTWalker()
        invocation_listener = InvocationTest()
        invocation_walker.walk(invocation_listener, ast)



    if not quiet:
        LOG_INFO("Errors: "+ str(errors))
        #print_parse_tree(tree, parser.ruleNames)
    return errors
 

if __name__ == '__main__':
    error_file_list = []
    if os.path.isdir(sys.argv[1]):
        filepath = sys.argv[1]
        files = os.listdir(filepath)
        files.sort()
        for fi in files:
            fpath = os.path.join(filepath, fi)
            LOG_INFO("======================== "+fi+" ========================")
            new_file, import_msg = Preprocess(fpath).import_process()
            input_stream = FileStream(new_file, encoding='utf-8')
            if main(input_stream)>0:
                error_file_list.append(fi)
            import_msg.clear_msg()

        LOG_INFO("======================== "+"error file result"+" ========================")
        for error_file in error_file_list:
             LOG_INFO(error_file)

    elif os.path.isfile(sys.argv[1]):
        new_file, import_msg = Preprocess(sys.argv[1]).import_process()
        input_stream = FileStream(new_file, encoding='utf-8')
        if main(input_stream)>0:
            LOG_INFO("======================== "+"error file result"+" ========================")
            LOG_INFO(sys.argv[1])
            pass
    else:
        pass

