from enum import Enum
from queue import Empty


class BaseType:
    def get_type_name(self):
        pass

    """
	It is useful during type computation and code gen to assign an int
	index to the primitive types and possibly user-defined types like
	structs and classes.
	@return Return 0-indexed type index else -1 if no index.
    """

    def get_type_index(self):
        pass


class PrimitiveType(BaseType):
    class Types(Enum):
        INVALID = 0
        INT = 1
        UINT = 2
        FLOAT = 3
        BOOL = 4
        STRING = 5

    def __init__(self):
        super().__init__()


class PhysicalType(BaseType):
    class Types(Enum):
        INVALID = 0
        KG = 1
        M = 2
        S = 3
        A = 4
        K = 5
        MOL = 6
        CD = 7
        FACTOR = 8
        OFFSET = 9
        RAD = 10

    def __init__(self):
        super().__init__()


class EnumType(BaseType):
    class Types(Enum):
        INVALID = 0
        UINT = 1
        HEX = 2
        Empty = 3

    def __init__(self, type=Types.INVALID, index=-1):
        super().__init__()
        self.elem_type = type
        self.elems_index = index

    def get_type_name(self):
        return self.elem_type

    def get_type_index(self):
        return self.elems_index
