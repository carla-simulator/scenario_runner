import os
import sys

from antlr4 import *
sys.path.append(os.getcwd())
try:
    sys.path.append('../')
except IndexError:
    pass

from srunner.osc2.osc_preprocess.pre_process import Preprocess
from srunner.osc2.utils.log_manager import *
from srunner.osc2.error_manager.error_listener import OscErrorListener
from srunner.osc2.osc2_parser.OpenSCENARIO2Lexer import OpenSCENARIO2Lexer
from srunner.osc2.osc2_parser.OpenSCENARIO2Parser import OpenSCENARIO2Parser
from srunner.osc2.ast_manager.ast_builder import ASTBuilder


 
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

    ast_listener = ASTBuilder()
    walker = ParseTreeWalker()
    walker.walk(ast_listener, tree)

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
            input_stream = FileStream(fpath, encoding='utf-8')
            if main(input_stream)>0:
                error_file_list.append(fi)
            import_msg.clear_msg()
        LOG_INFO("======================== "+"error file result"+" ========================")
        for error_file in error_file_list:
             LOG_INFO(error_file)

    elif os.path.isfile(sys.argv[1]):
        new_file, import_msg = Preprocess(sys.argv[1]).import_process()
        input_stream = FileStream(sys.argv[1], encoding='utf-8')
        if main(input_stream)>0:
            LOG_INFO("======================== "+"error file result"+" ========================")
            LOG_INFO(sys.argv[1])
            pass
    else:
        pass

