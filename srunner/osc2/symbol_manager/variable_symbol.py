from srunner.osc2.symbol_manager.typed_symbol import TypedSymbol


class VariableSymbol(TypedSymbol):
    def __init__(self, name, scope, type, value=None):
        super().__init__(name, scope, type, value)
