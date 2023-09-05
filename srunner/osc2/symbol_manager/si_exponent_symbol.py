from srunner.osc2.symbol_manager.base_symbol import BaseSymbol
from srunner.osc2.symbol_manager.local_scope import LocalScope


class SiExpSymbol(BaseSymbol):
    def __init__(self, name, value, scope):
        super().__init__(name, scope)
        self.value = value

    def __str__(self):
        buf = self.__class__.__name__
        buf += " : "
        buf += self.name
        return buf


class SiBaseExponentListScope(LocalScope):
    def __init__(self, scope):
        super().__init__(scope)

    def __str__(self):
        return "()"
