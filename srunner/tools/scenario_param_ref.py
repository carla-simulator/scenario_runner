#!/usr/bin/env python

"""
This module provides a class for retrieving parameter values of
scenario configuration files based on OpenSCENARIO
"""

import re
from distutils.util import strtobool

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider


class ParameterRef:
    """
    This class stores osc parameter reference in its original form.
    Returns the converted value whenever it is used.
    """

    def __init__(self, reference_text) -> None:
        # TODO: (for OSC1.1) add methods(lexer and math_interpreter) to
        #  recognize and interpret math expression from reference_text
        self.reference_text = str(reference_text)

    def is_literal(self) -> bool:
        """
        Returns: True when text is a literal/number
        """
        return self._is_matching(pattern=r"(-)?\d+(\.\d*)?")

    def is_parameter(self) -> bool:
        """
        Returns: True when text is a parameter
        """
        return self._is_matching(pattern=r"[$][A-Za-z_][\w]*")

    def _is_matching(self, pattern: str) -> bool:
        """
        Returns: True when pattern is matching with text
        """
        match = re.search(pattern, self.reference_text)
        if match is not None:
            matching_string = match.group()
            return matching_string == self.reference_text
        return False

    def _cast_type(self, other):
        try:
            if isinstance(other, str):
                return self.__str__()
            elif isinstance(other, float):
                return self.__float__()
            elif isinstance(other, int):
                return self.__int__()
            elif isinstance(other, bool):
                return self.__bool__()
            else:
                raise Exception("Type conversion for type {} not implemented".format(type(other)))
        except ValueError:
            # If the parameter value can't be converted into type of 'other'
            return None

    def get_interpreted_value(self):
        """
        Returns: interpreted value from reference_text
        """
        if self.is_literal():
            value = self.reference_text
        elif self.is_parameter():
            value = CarlaDataProvider.get_osc_global_param_value(self.reference_text)
            if value is None:
                raise Exception("Parameter '{}' is not defined".format(self.reference_text[1:]))
        else:
            value = None
        return value

    def __float__(self) -> float:
        value = self.get_interpreted_value()
        if value is not None:
            return float(value)
        else:
            raise Exception("could not convert '{}' to float".format(self.reference_text))

    def __int__(self) -> int:
        value = self.get_interpreted_value()
        if value is not None:
            return int(float(value))
        else:
            raise Exception("could not convert '{}' to int".format(self.reference_text))

    def __str__(self) -> str:
        value = self.get_interpreted_value()
        return str(value) if value is not None else self.reference_text

    def __bool__(self) -> bool:
        try:
            return bool(strtobool(self.__str__()))
        except ValueError:
            raise Exception("could not convert '{}' to bool".format(self.__str__()))

    def __repr__(self):
        value = self.get_interpreted_value()
        return value if value is not None else self.reference_text

    def __hash__(self):
        value = self.get_interpreted_value()
        return hash(value) if value is not None else hash(self.reference_text)

    def __radd__(self, other) -> bool:
        return other + self.__float__()

    def __add__(self, other) -> bool:
        return other + self.__float__()

    def __rsub__(self, other) -> bool:
        return other - self.__float__()

    def __sub__(self, other) -> bool:
        return self.__float__() - other

    def __rmul__(self, other) -> bool:
        return other * self.__float__()

    def __mul__(self, other) -> bool:
        return other * self.__float__()

    def __truediv__(self, other) -> bool:
        return self.__float__() / other

    def __rtruediv__(self, other) -> bool:
        return other / self.__float__()

    def __eq__(self, other) -> bool:
        value = self._cast_type(other)
        return other == value

    def __ne__(self, other) -> bool:
        value = self._cast_type(other)
        return other != value

    def __ge__(self, other) -> bool:
        value = self._cast_type(other)
        return value >= other

    def __le__(self, other) -> bool:
        value = self._cast_type(other)
        return value <= other

    def __gt__(self, other) -> bool:
        value = self._cast_type(other)
        return value > other

    def __lt__(self, other) -> bool:
        value = self._cast_type(other)
        return value < other

    def __GE__(self, other) -> bool:  # pylint: disable=invalid-name
        value = self._cast_type(other)
        return value >= other

    def __LE__(self, other) -> bool:  # pylint: disable=invalid-name
        value = self._cast_type(other)
        return value <= other

    def __GT__(self, other) -> bool:  # pylint: disable=invalid-name
        value = self._cast_type(other)
        return value > other

    def __LT__(self, other) -> bool:  # pylint: disable=invalid-name
        value = self._cast_type(other)
        return value < other

    def __iadd__(self, other) -> bool:
        return self.__float__() + other

    def __isub__(self, other) -> bool:
        return self.__float__() - other

    def __abs__(self):
        return abs(self.__float__())
