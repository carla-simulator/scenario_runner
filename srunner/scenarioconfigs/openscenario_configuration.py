#!/usr/bin/env python

# Copyright (c) 2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides the key configuration parameters for a scenario based on OpenSCENARIO
"""

from __future__ import print_function

import math
import os
import xml.etree.ElementTree as ET

import xmlschema

import carla

# pylint: disable=line-too-long
from srunner.scenarioconfigs.scenario_configuration import ActorConfigurationData, ScenarioConfiguration, WeatherConfiguration
# pylint: enable=line-too-long
from srunner.tools.openscenario_parser import OpenScenarioParser


class OpenScenarioConfiguration(ScenarioConfiguration):

    """
    Limitations:
    - Only one Story + Init is supported per Storyboard
    """

    def __init__(self, filename):

        self.xml_tree = ET.parse(filename)

        self._set_global_parameters()
        self._validate_openscenario_configuration()

        self.other_actors = []
        self.ego_vehicles = []
        self.trigger_points = []
        self.weather = WeatherConfiguration()

        self.storyboard = self.xml_tree.find("Storyboard")
        self.story = self.storyboard.find("Story")
        self.init = self.storyboard.find("Init")

        self._parse_openscenario_configuration()

    def _validate_openscenario_configuration(self):
        """
        Validate the given OpenSCENARIO config against the 0.9.1 XSD

        Note: This will throw if the config is not valid. But this is fine here.
        """
        xsd_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../openscenario/OpenSCENARIO_v0.9.1.xsd")
        xsd = xmlschema.XMLSchema(xsd_file)
        xsd.validate(self.xml_tree)

    def _parse_openscenario_configuration(self):
        """
        Parse the given OpenSCENARIO config file, set and validate parameters
        """
        self._set_scenario_name()
        self._set_carla_town()
        self._set_actor_information()
        self._set_carla_weather()
        self._set_carla_friction()

        self._validate_result()

    def _set_scenario_name(self):
        """
        Extract the scenario name from the OpenSCENARIO header information
        """
        header = self.xml_tree.find("FileHeader")
        self.name = header.attrib.get('description', 'Unknown')

        if self.name.startswith("CARLA:"):
            OpenScenarioParser.set_use_carla_coordinate_system()

    def _set_carla_town(self):
        """
        Extract the CARLA town (level) from the RoadNetwork information from OpenSCENARIO

        Note: The specification allows multiple Logics elements within the RoadNetwork element.
              Hence, there can be multiple towns specified. We just use the _last_ one.
        """
        for logic in self.xml_tree.find("RoadNetwork").findall("Logics"):
            self.town = logic.attrib.get('filepath', None)

        if self.town is not None and ".xodr" in self.town:
            (_, tail) = os.path.split(self.town)
            self.town = tail[:-5]

    def _set_carla_weather(self):
        """
        Extract weather information from OpenSCENARIO config
        """

        for weather in self.init.iter("Weather"):
            sun = weather.find("Sun")
            self.weather.sun_azimuth = math.degrees(float(sun.attrib.get('azimuth', 0)))
            self.weather.sun_altitude = math.degrees(float(sun.attrib.get('elevation', 0)))
            self.weather.cloudyness = 100 - float(sun.attrib.get('intensity', 0)) * 100
            precepitation = weather.find("Precipitation")
            self.weather.precipitation = 0
            if precepitation.attrib.get('type') == "rain":
                self.weather.precipitation = float(precepitation.attrib.get('intensity')) * 100
                self.weather.precipitation_deposits = 100  # if it rains, make the road wet
            elif precepitation.attrib.get('type') == "snow":
                raise AttributeError("CARLA does not support snow precipitation")

    def _set_carla_friction(self):
        """
        Extract road friction information from OpenSCENARIO config
        """

        road_condition = self.init.iter("RoadCondition")
        for condition in road_condition:
            self.friction = float(condition.attrib.get('frictionScale'))

    def _set_global_parameters(self):
        """
        Parse the complete scenario definition file, and replace all global parameter references
        with the actual values
        """

        global_parameters = dict()
        parameters = self.xml_tree.find('ParameterDeclaration')

        if parameters is None:
            return

        for parameter in parameters:
            name = parameter.attrib.get('name')
            value = parameter.attrib.get('value')

            global_parameters[name] = value

        for node in self.xml_tree.find('Entities').iter():
            for key in node.attrib:
                for param in global_parameters:
                    if node.attrib[key] == param:
                        node.attrib[key] = global_parameters[param]

        for node in self.xml_tree.find('Storyboard').iter():
            for key in node.attrib:
                for param in global_parameters:
                    if node.attrib[key] == param:
                        node.attrib[key] = global_parameters[param]

    def _set_actor_information(self):
        """
        Extract all actors and their corresponding specification

        NOTE: The rolename property has to be unique!
        """
        for entity in self.xml_tree.iter("Entities"):
            for obj in entity.iter("Object"):
                rolename = obj.attrib.get('name', 'simulation')
                for vehicle in obj.iter("Vehicle"):
                    color = None
                    model = vehicle.attrib.get('name', "vehicle.*")
                    category = vehicle.attrib.get('category', "car")
                    ego_vehicle = False
                    for prop in obj.iter("Property"):
                        if prop.get('name', '') == 'type':
                            ego_vehicle = prop.get('value') == 'ego_vehicle'
                        if prop.get('name', '') == 'color':
                            color = prop.get('value')

                    new_actor = ActorConfigurationData(
                        model, carla.Transform(), rolename, color=color, category=category)
                    new_actor.transform = self._get_actor_transform(rolename)

                    if ego_vehicle:
                        self.ego_vehicles.append(new_actor)
                    else:
                        self.other_actors.append(new_actor)

                for pedestrian in obj.iter("Pedestrian"):
                    model = pedestrian.attrib.get('model', "walker.*")

                    new_actor = ActorConfigurationData(model, carla.Transform(), rolename)
                    new_actor.transform = self._get_actor_transform(rolename)

                    self.other_actors.append(new_actor)

    def _get_actor_transform(self, actor_name):
        """
        Get the initial actor transform provided by the Init section

        Note: - The OpenScenario specification allows multiple definitions. We use the _first_ one
              - The OpenScenario specification allows different ways of specifying a position.
                We currently only support a specification with absolute world coordinates
        """

        actor_transform = carla.Transform()

        actor_found = False

        for private_action in self.init.iter("Private"):
            if private_action.attrib.get('object', None) == actor_name:
                if actor_found:
                    # pylint: disable=line-too-long
                    print(
                        "Warning: The actor '{}' was already assigned an initial position. Overwriting pose!".format(actor_name))
                    # pylint: enable=line-too-long
                actor_found = True
                for position in private_action.iter('Position'):
                    transform = OpenScenarioParser.convert_position_to_transform(position)
                    if transform:
                        actor_transform = transform

        if not actor_found:
            print("Warning: The actor '{}' was not assigned an initial position. Using (0,0,0)".format(actor_name))

        return actor_transform

    def _validate_result(self):
        """
        Check that the current scenario configuration is valid
        """
        if not self.name:
            raise AttributeError("No scenario name found")

        if not self.town:
            raise AttributeError("CARLA level not defined")

        if not self.ego_vehicles:
            raise AttributeError("No ego vehicles defined in scenario")
