from srunner.osc2.symbol_manager.typed_symbol import TypedSymbol


class ParameterSymbol(TypedSymbol):
    def __init__(self, name, scope, type, value=None):
        super().__init__(name, scope, type, value)

    def __str__(self):
        buf = self.type
        buf += ":"
        buf += self.name
        if self.value is not None:
            buf += ","
            buf += str(self.value)
        return buf
