from __future__ import print_function

import math
import operator
from typing import List, Tuple

import carla
import numpy as np
from antlr4.CommonTokenStream import CommonTokenStream
from antlr4.FileStream import FileStream
from antlr4.tree.Tree import ParseTreeWalker
from numpy.linalg import det

from srunner.osc2.ast_manager.ast_builder import ASTBuilder
from srunner.osc2.error_manager.error_listener import OscErrorListener
from srunner.osc2.osc2_parser.OpenSCENARIO2Lexer import OpenSCENARIO2Lexer as OSC2Lexer
from srunner.osc2.osc2_parser.OpenSCENARIO2Parser import (
    OpenSCENARIO2Parser as OSC2Parser,
)
from srunner.osc2.osc_preprocess.pre_process import Preprocess


class OSC2Helper(object):
    osc2_file = None
    ast_tree = None
    ego_name = "ego_vehicle"
    wait_for_ego = False

    @classmethod
    def gen_osc2_ast(cls, osc2_file_name: str):
        if osc2_file_name == cls.osc2_file:
            return cls.ast_tree
        else:
            # preprocessing
            new_file, _ = Preprocess(osc2_file_name).import_process()
            input_stream = FileStream(new_file, encoding="utf-8")

            osc_error_listeners = OscErrorListener(input_stream)
            lexer = OSC2Lexer(input_stream)
            lexer.removeErrorListeners()
            lexer.addErrorListener(osc_error_listeners)

            tokens = CommonTokenStream(lexer)
            parser = OSC2Parser(tokens)
            parser.removeErrorListeners()
            parser.addErrorListener(osc_error_listeners)
            parse_tree = parser.osc_file()

            osc2_ast_builder = ASTBuilder()
            walker = ParseTreeWalker()
            walker.walk(osc2_ast_builder, parse_tree)

            cls.ast_tree = osc2_ast_builder.get_ast()

        return cls.ast_tree

    @staticmethod
    def vector_angle(v1: List[int], v2: List[int]) -> int:
        """Calculate the angle between vectors v1 and v2.
        Parameters:
            v1: vector v1, list type [x1, y1, x2, y2], where x1.y1 represents the starting coordinates and x2, y2 represent the ending coordinates.
            v2: same as above.
        Return: the angle, positive for clockwise and negative for counterclockwise.
        """
        dx1 = v1[2] - v1[0]
        dy1 = v1[3] - v1[1]
        dx2 = v2[2] - v2[0]
        dy2 = v2[3] - v2[1]
        angle1 = math.atan2(dy1, dx1)
        angle1 = int(math.degrees(angle1))
        angle2 = math.atan2(dy2, dx2)
        angle2 = int(math.degrees(angle2))
        return angle1 - angle2

    ###########################################################################
    # Determine the center and radius of a circle from three points on the circle
    ###########################################################################
    # INPUT
    # p1   :  - coordinates of the first point, list or array 1x3
    # p2   :  - coordinates of the second point, list or array 1x3
    # p3   :  - coordinates of the third point, list or array 1x3
    # If a 1x2 row vector is input, the last element is automatically filled with 0 to become a 1x3 row vector.
    ###########################################################################
    # OUTPUT
    # pc   :  - coordinates of the center of the circle, array 1x3
    # r    :  - radius, scalar
    ###########################################################################
    # Example 1 - three points on a plane
    # pc1, r1 = points2circle([1, 2], [-2, 1], [0, -3])
    # Example 2 - three points in space
    # pc2, r2 = points2circle([1, 2, -1], [-2, 1, 2], [0, -3, -3]).
    ###########################################################################
    @staticmethod
    def curve_radius(p1: List, p2: List, p3: List) -> Tuple:
        p1 = np.array(p1)
        p2 = np.array(p2)
        p3 = np.array(p3)
        num1 = len(p1)
        num2 = len(p2)
        num3 = len(p3)

        # check inputs
        if (num1 == num2) and (num2 == num3):
            if num1 == 2:
                p1 = np.append(p1, 0)
                p2 = np.append(p2, 0)
                p3 = np.append(p3, 0)
            elif num1 != 3:
                return None
        else:
            return None

        # Collinearity check
        temp01 = p1 - p2
        temp02 = p3 - p2
        temp03 = np.cross(temp01, temp02)
        temp = (temp03 @ temp03) / (temp01 @ temp01) / (temp02 @ temp02)
        if temp < 10**-6:
            return None

        temp1 = np.vstack((p1, p2, p3))
        temp2 = np.ones(3).reshape(3, 1)
        mat1 = np.hstack((temp1, temp2))  # size = 3x4

        m = +det(mat1[:, 1:])
        n = -det(np.delete(mat1, 1, axis=1))
        p = +det(np.delete(mat1, 2, axis=1))
        q = -det(temp1)

        temp3 = np.array([p1 @ p1, p2 @ p2, p3 @ p3]).reshape(3, 1)
        temp4 = np.hstack((temp3, mat1))
        temp5 = np.array([2 * q, -m, -n, -p, 0])
        mat2 = np.vstack((temp4, temp5))  # size = 4x5

        A = +det(mat2[:, 1:])
        B = -det(np.delete(mat2, 1, axis=1))
        C = +det(np.delete(mat2, 2, axis=1))
        D = -det(np.delete(mat2, 3, axis=1))
        E = +det(mat2[:, :-1])

        pc = -np.array([B, C, D]) / 2 / A
        r = np.sqrt(B * B + C * C + D * D - 4 * A * E) / 2 / abs(A)
        _ = list(map(lambda n: round(n, 2), pc.tolist()))

        return round(r, 2)

    @staticmethod
    def point_line_location(A: Tuple, B: Tuple, C: Tuple) -> str:
        """
        The position of point C relative to directed line segment AB.
        return: string, left, on, right
        """
        xa = A[0]
        ya = A[1]
        xb = B[0]
        yb = B[1]
        xc = C[0]
        yc = C[1]
        f = (xb - xa) * (yc - ya) - (xc - xa) * (yb - ya)
        if f > 0:
            return "left"
        elif f == 0:
            return "on"
        else:
            return "right"

    @staticmethod
    def find_physical_type(src_si_base_exponent: dict, si_base_exponent: dict):
        for key, value in src_si_base_exponent.items():
            if operator.eq(value.si_base_exponent, si_base_exponent):
                return key
        return None

    @staticmethod
    def euler_orientation(rotation: carla.Rotation):
        pitch = rotation.pitch
        yaw = rotation.yaw
        roll = rotation.roll
        x = math.sin(pitch / 2) * math.sin(yaw / 2) * math.cos(roll / 2) + math.cos(
            pitch / 2
        ) * math.cos(yaw / 2) * math.sin(roll / 2)
        y = math.sin(pitch / 2) * math.cos(yaw / 2) * math.cos(roll / 2) + math.cos(
            pitch / 2
        ) * math.sin(yaw / 2) * math.sin(roll / 2)
        z = math.cos(pitch / 2) * math.sin(yaw / 2) * math.cos(roll / 2) - math.sin(
            pitch / 2
        ) * math.cos(yaw / 2) * math.sin(roll / 2)
        w = math.cos(pitch / 2) * math.cos(yaw / 2) * math.cos(roll / 2) - math.sin(
            pitch / 2
        ) * math.sin(yaw / 2) * math.sin(roll / 2)
        return x, y, z, w

    @staticmethod
    def flat_list(list_of_lists):
        if len(list_of_lists) == 0:
            return list_of_lists

        if isinstance(list_of_lists[0], list):
            return OSC2Helper.flat_list(list_of_lists[0]) + OSC2Helper.flat_list(
                list_of_lists[1:]
            )

        return list_of_lists[:1] + OSC2Helper.flat_list(list_of_lists[1:])
