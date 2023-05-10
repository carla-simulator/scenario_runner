from math import exp
import os
import sys
from os.path import abspath, dirname
from typing import Tuple
from antlr4 import *

try:
    sys.path.append('../')
except IndexError:
    pass

from srunner.osc2.ast_manager.ast_node import AST, Expression
import srunner.osc2.ast_manager.ast_node as ast_node

from srunner.osc2.utils.log_manager import *
from srunner.osc2.utils.tools import *
from srunner.osc2.error_manager.error_listener import OscErrorListener
from srunner.osc2.osc2_parser.OpenSCENARIO2Lexer import OpenSCENARIO2Lexer
from srunner.osc2.osc2_parser.OpenSCENARIO2Parser import OpenSCENARIO2Parser
from srunner.osc2.ast_manager.ast_builder import ASTBuilder

from srunner.osc2.ast_manager.ast_vistor import ASTVisitor
from srunner.osc2.osc_preprocess.pre_process import ImportFile, Preprocess
import graphviz

# node：入参，树的节点
# nodes：入参，按遍历的顺序给节点编号，用编号给节点连线
# pindex：父节点的编号
# g：graphviz的图对象
def render_ast(node, nodes, pindex, g):
    if not isinstance(node, Tuple):
        name = str(node)

        index = len(nodes)
        nodes.append(index)

        g.node(str(index), name)
        if index != pindex:
            # g.edge(str(index), str(pindex))
            g.edge(str(pindex), str(index)) # 边是从父到子，要不会出现倒立的树
        if isinstance(node, ast_node.AST):
            for i in range(0, node.get_child_count()):
                render_ast(node.get_child(i), nodes, index, g)

operator = []
num = []
all = []
def traverse_list(list_name):
    for ex in list_name:
        if isinstance(ex, list):
            traverse_list(ex)
        else:
            if ex == '>' or ex == '>=' or ex == '==' or ex == '+' or ex == 'in':
                operator.append(ex)
            else:
                num.append(ex)
            all.append(ex)
            
                    



class InvocationTest(ASTVisitor):

    def visit_modifier_invocation(self,  node: ast_node.ModifierInvocation):
        print("visit modifier invocation!")
        print("modifier name:", node.modifier_name)
        arguments = self.visit_children(node)
        print(arguments)
    
    def visit_method_body(self, node: ast_node.MethodBody):
        print("visit method body!")
        print("method body name:", node.type)
        arguments = self.visit_children(node)
        print(arguments)
        traverse_list(arguments)
        print(num, operator, all)
        tempStack = []
        for ex in all:
            if ex == '>' or ex == '>=' or ex == '==' or ex == '+':
                right = tempStack.pop()
                left = tempStack.pop()
                expression = left + ' ' + ex + ' ' + right
                tempStack.append(expression)
            elif ex == 'in':
                right = tempStack.pop()
                left = tempStack.pop()
                opnum = tempStack.pop()
                expression = opnum + ' ' + ex + ' [' + left + ', '+ right + ']'
                tempStack.append(expression)
            else:
                tempStack.append(ex)
        print(tempStack.pop())
        operator.clear()
        num.clear()
        all.clear()
    
    def visit_relation_expression(self, node: ast_node.RelationExpression):
        return [self.visit_children(node), node.operator]

    def visit_binary_expression(self, node: ast_node.BinaryExpression):
        return [self.visit_children(node), node.operator]

    def visit_named_argument(self,  node: ast_node.NamedArgument):
        return node.argument_name, self.visit_children(node)

    def visit_physical_literal(self,  node: ast_node.PhysicalLiteral):
        return node.value, node.unit_name

    def visit_integer_literal(self,  node: ast_node.IntegerLiteral):
        return node.value

    def visit_float_literal(self,  node: ast_node.FloatLiteral):
        return node.value

    def visit_bool_literal(self,  node: ast_node.BoolLiteral):
        return node.value

    def visit_string_literal(self,  node: ast_node.StringLiteral):
        return node.value

    def visit_identifier(self,  node: ast_node.Identifier):
        return node.name



 
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

        #graph = graphviz.Graph(node_attr={'shape': 'plaintext'},format='png')
        #render_ast(ast, [], 0, graph)
        #graph.view()

        invocation_visitor = InvocationTest()
        invocation_visitor.visit(ast)



    if not quiet:
        LOG_INFO("Errors: "+ str(errors))
        #print_parse_tree(tree, parser.ruleNames)
    return errors
 

if __name__ == '__main__':
    print("执行预处理")
    error_file_list = []
    # 如果测试的为文件夹
    if not os.path.exists(sys.argv[1]):
        print("文件路径错误！")
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

    # 如果测试的为单个文件
    elif os.path.isfile(sys.argv[1]):
        print("执行预处理")
        # 预处理，展开import,返回file类型对象和存储预处理信息的对象
        new_file, import_msg = Preprocess(sys.argv[1]).import_process()
        input_stream = FileStream(new_file, encoding='utf-8')
        if main(input_stream)>0:
            LOG_INFO("======================== "+"error file result"+" ========================")
            LOG_INFO(sys.argv[1])
            pass
    else:
        pass

