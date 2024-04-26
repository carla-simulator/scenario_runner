import operator
from enum import Enum


class RelationalOperator(Enum):
    EQUALITY = "=="
    INEQUALITY = "!="
    LESS_THAN = "<"
    LESS_OR_EQUAL = "<="
    GREATER_THAN = ">"
    GREATER_OR_EQUAL = ">="
    MEMBERSHIP = "in"

    @staticmethod
    def values():
        return [member.value for member in RelationalOperator]
