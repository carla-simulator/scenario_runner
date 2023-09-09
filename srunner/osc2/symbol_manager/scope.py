# A scope is a dictionary of symbols that are grouped together by some
# lexical construct in the input language. Examples include structs,
# functions, {...} code blocks, argument lists, etc...

from srunner.osc2.utils.log_manager import LOG_ERROR


class Scope:
    # Gets the scope domain name
    def get_scope_name(self):
        raise BaseException("get_scope_name must be overload!")

    # Find the parent scope, returning None if the parent scope is empty, such as GlobalScope
    def get_enclosing_scope(self):
        pass

    # Defines symbols in the current scope
    def define(self, sym):
        pass

    # Look up variable names in the symbol table recursively, starting at the current scope
    def resolve(self, name):
        pass

    # Gets all symbol table names in the current scope
    def get_symbol_names():
        pass

    def get_child_symbol(self, i):
        raise BaseException("get_child_symbol must be overload!")

    # Gets the number of symbol tables in the current scope
    def get_number_of_symbols():
        raise BaseException("get_number_of_symbols must be overload!")

    # Gets the number of symbol tables in the current scope
    def get_enclosing_path_to_root():
        pass
