from srunner.osc2.symbol_manager.base_symbol import BaseSymbol
from srunner.osc2.utils.log_manager import *
from srunner.osc2.symbol_manager.si_exponent_symbol import SiExpSymbol


class PhysicalTypeSymbol(BaseSymbol):

    def __init__(self, name, scope):
        super().__init__(name, scope)
