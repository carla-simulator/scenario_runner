from srunner.osc2.symbol_manager.base_symbol import BaseSymbol


class WaitSymbol(BaseSymbol):
    def __init__(self, scope):
        super().__init__("wait", scope)
