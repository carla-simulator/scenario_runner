# A generic programming language symbol. A symbol has to have a name and
# a scope in which it lives. It also helps to know the order in which
# symbols are added to a scope because this often translates to
# register or parameter numbers.


class Symbol:
    def __init__(self, name=None, scope=None):
        self.name = name
        self.scope = scope

    def get_scope_name(self):
        return self.name

    def __str__(self) -> str:
        return self.__class__.__name__
