from srunner.osc2.symbol_manager.base_symbol import BaseSymbol
from srunner.osc2.symbol_manager.typed_symbol import TypedSymbol


class ArgumentSymbol(BaseSymbol):
    def __init__(self, name, scope):
        super().__init__(name, scope)


class NamedArgumentSymbol(ArgumentSymbol):
    def __init__(self, name, scope):
        super().__init__(name, scope)

    def __str__(self):
        buf = self.__class__.__name__
        buf += " : "
        buf += self.name
        return buf


class PositionalArgumentSymbol(ArgumentSymbol):
    def __init__(self, scope):
        super().__init__(None, scope)

    def __str__(self):
        buf = self.__class__.__name__
        return buf


class ArgumentSpecificationSymbol(TypedSymbol):
    def __init__(self, name, scope, type, value=None):
        super().__init__(name, scope, type, value)

    def __str__(self):
        buf = self.__class__.__name__
        buf += " : "
        buf += self.type
        buf += ","
        buf += self.name
        if self.value is not None:
            buf += ","
            buf += str(self.value)
        return buf
