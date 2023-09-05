from srunner.osc2.symbol_manager.base_symbol import BaseSymbol
from srunner.osc2.utils.log_manager import LOG_ERROR


class UnitSymbol(BaseSymbol):
    def __init__(self, name, scope, physical_name):
        super().__init__(name, scope)
        self.physical_name = physical_name
