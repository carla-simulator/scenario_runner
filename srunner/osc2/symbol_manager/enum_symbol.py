from srunner.osc2.error_manager import *
from srunner.osc2.symbol_manager.base_symbol import BaseSymbol
from srunner.osc2.utils.log_manager import LOG_ERROR


class EnumValueRefSymbol(BaseSymbol):
    def __init__(self, enum_name, enum_member_name, value, scope):
        name = enum_name + "!" + enum_member_name
        super().__init__(name, scope)
        self.enum_name = enum_name
        self.enum_member_name = enum_member_name
        self.value = value

    def __str__(self):
        buf = self.name
        buf += " = "
        buf += str(self.value)
        return buf


class EnumMemberSymbol(BaseSymbol):
    def __init__(self, name, scope, member_value=0):
        super().__init__(name, scope)
        self.elems_index = member_value

    def __str__(self):
        buf = super().__str__()
        buf += " : "
        buf += str(self.elems_index)
        return buf


class EnumSymbol(BaseSymbol):
    def __init__(self, name, scope):
        super().__init__(name, scope)
        self.last_index = -1

    def define(self, sym: EnumMemberSymbol, ctx):
        # Check for name conflicts
        if sym.name in self.symbols and sym.name:
            msg = "Enum member '" + sym.name + "' is already defined!"
            LOG_ERROR(msg, ctx)

        # Check that the enumeration element value is correct
        # The default value of an enumeration member must be a subsequent integer value
        if (
            sym.elems_index != 0
            and sym.elems_index != self.last_index + 1
            and self.last_index != -1
        ):
            msg = (
                "Enum member '"
                + sym.name
                + "' with wrong Value: "
                + str(sym.elems_index)
            )
            LOG_ERROR(msg, ctx)
        else:
            self.symbols[sym.name] = sym
            self.last_index = sym.elems_index

    def __str__(self):
        buf = "enum "
        buf += super().__str__()
        return buf
