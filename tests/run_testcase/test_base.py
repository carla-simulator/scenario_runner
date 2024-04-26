import os
import sys

from srunner.osc2.ast_manager.ast_builder import ASTBuilder
from srunner.osc2.osc2_parser.OpenSCENARIO2Lexer import OpenSCENARIO2Lexer
from srunner.osc2.osc2_parser.OpenSCENARIO2Parser import OpenSCENARIO2Parser

from antlr4 import *
from srunner.osc2.osc_preprocess.pre_process import Preprocess
from tests.run_testcase.log_msg import create_LogMsg as log_msg
from srunner.osc2.osc_preprocess.import_msg import create_ImportMsg as import_msg


# class that stores error information


# basic class used for testing
# From inputting grammar files to generating AST
# return True
class TestASTClass(object):
    def main(self, input_stream):
        lexer = OpenSCENARIO2Lexer(input_stream)
        stream = CommonTokenStream(lexer)

        parser = OpenSCENARIO2Parser(stream)
        tree = parser.osc_file()
        listener = ParseTreeWalker()
        ast_builder = ASTBuilder()
        listener.walk(ast_builder, tree)
        import_msg.clear_msg()
        return True

    def testcase(self, str):
        new_file, import_msg = Preprocess(str).import_process()
        input_stream = FileStream(new_file, encoding='utf-8')
        return self.main(input_stream)


# Class for testing symbol tables
class TestSymbolClass(object):
    def main(self, input_stream):
        lexer = OpenSCENARIO2Lexer(input_stream)
        stream = CommonTokenStream(lexer)

        parser = OpenSCENARIO2Parser(stream)
        tree = parser.osc_file()
        listener = ParseTreeWalker()
        ast_builder = ASTBuilder()
        listener.walk(ast_builder, tree)
        import_msg.clear_msg()
        return log_msg.get_log_msg()

    def testcase(self, str):
        new_file, import_msg = Preprocess(str).import_process()
        input_stream = FileStream(new_file, encoding='utf-8')
        return self.main(input_stream)
