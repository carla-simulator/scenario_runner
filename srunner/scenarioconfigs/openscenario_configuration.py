#!/usr/bin/env python

# Copyright (c) 2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides the key configuration parameters for a scenario based on OpenSCENARIO
"""

from __future__ import print_function

import logging
import math
import os
import xml.etree.ElementTree as ET

import xmlschema

import carla

# pylint: disable=line-too-long
from srunner.scenarioconfigs.scenario_configuration import ActorConfigurationData, ScenarioConfiguration, WeatherConfiguration
# pylint: enable=line-too-long
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider  # workaround
from srunner.tools.openscenario_parser import OpenScenarioParser


class OpenScenarioConfiguration(ScenarioConfiguration):

    """
    Limitations:
    - Only one Story + Init is supported per Storyboard
    """

    def __init__(self, filename, client):

        self.xml_tree = ET.parse(filename)

        self._set_global_parameters()
        self._validate_openscenario_configuration()
        self.client = client

        self.catalogs = {}

        self.other_actors = []
        self.ego_vehicles = []
        self.trigger_points = []
        self.weather = WeatherConfiguration()

        self.storyboard = self.xml_tree.find("Storyboard")
        self.story = self.storyboard.find("Story")
        self.init = self.storyboard.find("Init")

        logging.basicConfig()
        self.logger = logging.getLogger("OpenScenarioConfiguration")

        self._parse_openscenario_configuration()

    def _validate_openscenario_configuration(self):
        """
        Validate the given OpenSCENARIO config against the 0.9.1 XSD

        Note: This will throw if the config is not valid. But this is fine here.
        """
        xsd_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../openscenario/OpenSCENARIO_v0.9.1.xsd")
        xsd = xmlschema.XMLSchema(xsd_file)
        xsd.validate(self.xml_tree)

    def _validate_openscenario_catalog_configuration(self, catalog_xml_tree):
        """
        Validate the given OpenSCENARIO catalog config against the 0.9.1 XSD

        Note: This will throw if the catalog config is not valid. But this is fine here.
        """
        xsd_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../openscenario/OpenSCENARIO_Catalog.xsd")
        xsd = xmlschema.XMLSchema(xsd_file)
        xsd.validate(catalog_xml_tree)

    def _parse_openscenario_configuration(self):
        """
        Parse the given OpenSCENARIO config file, set and validate parameters
        """
        self._load_catalogs()

        self._set_scenario_name()
        self._set_carla_town()
        self._set_actor_information()
        self._set_carla_weather()
        self._set_carla_friction()

        self._validate_result()

    def _load_catalogs(self):
        """
        Read Catalog xml files into dictionary for later use

        NOTE: Catalogs must have distinct names, even across different types
        """
        catalogs = self.xml_tree.find("Catalogs")
        catalog_types = ["Vehicle",
                         "Driver",
                         "Pedestrian",
                         "PedestrianController",
                         "MiscObject",
                         "Environment",
                         "Maneuver",
                         "Route"]
        for catalog_type in catalog_types:
            catalog_path = catalogs.find(catalog_type + "Catalog") \
                                   .find("Directory") \
                                   .attrib.get('path')
            if not os.path.isfile(catalog_path):
                self.logger.warning("The %s path for the %s Catalog is invalid", catalog_path, catalog_type)
            else:
                xml_tree = ET.parse(catalog_path)
                self._validate_openscenario_catalog_configuration(xml_tree)
                catalog = xml_tree.find("Catalog")
                catalog_name = catalog.attrib.get("name")
                self.catalogs[catalog_name] = {}
                for entry in catalog:
                    self.catalogs[catalog_name][entry.attrib.get("name")] = entry

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

        # workaround for relative positions during init
        world = self.client.get_world()
        if world is None or world.get_map().name != self.town:
            self.client.load_world(self.town)
            world = self.client.get_world()
            CarlaDataProvider.set_world(world)
            world.wait_for_tick()

    def _set_carla_weather(self):
        """
        Extract weather information from OpenSCENARIO config
        """

        set_environment = next(self.init.iter("SetEnvironment"))

        if sum(1 for _ in set_environment.iter("Weather")) != 0:
            environment = set_environment.find("Environment")
        elif set_environment.find("CatalogReference") is not None:
            catalog_reference = set_environment.find("CatalogReference")
            environment = self.catalogs[catalog_reference.attrib.get(
                "catalogName")][catalog_reference.attrib.get("entryName")]

        weather = environment.find("Weather")
        sun = weather.find("Sun")
        self.weather.sun_azimuth = math.degrees(float(sun.attrib.get('azimuth', 0)))
        self.weather.sun_altitude = math.degrees(float(sun.attrib.get('elevation', 0)))
        self.weather.cloudiness = 100 - float(sun.attrib.get('intensity', 0)) * 100
        fog = weather.find("Fog")
        self.weather.fog_distance = float(fog.attrib.get('visualRange', 'inf'))
        if self.weather.fog_distance < 1000:
            self.weather.fog_density = 100
        self.weather.precipitation = 0
        self.weather.precipitation_deposits = 0
        self.weather.wetness = 0
        self.weather.wind_intensity = 0
        precepitation = weather.find("Precipitation")
        if precepitation.attrib.get('type') == "rain":
            self.weather.precipitation = float(precepitation.attrib.get('intensity')) * 100
            self.weather.precipitation_deposits = 100  # if it rains, make the road wet
            self.weather.wetness = self.weather.precipitation
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
                args = dict()
                for prop in obj.iter("Property"):
                    key = prop.get('name')
                    value = prop.get('value')
                    args[key] = value

                for catalog_reference in obj.iter("CatalogReference"):
                    entry = self.catalogs[catalog_reference.attrib.get(
                        "catalogName")][catalog_reference.attrib.get("entryName")]
                    if entry.tag == "Vehicle":
                        self._extract_vehicle_information(entry, rolename, entry, args)
                    elif entry.tag == "Pedestrian":
                        self._extract_pedestrian_information(entry, rolename, entry, args)
                    elif entry.tag == "MiscObject":
                        self._extract_misc_information(entry, rolename, entry, args)
                    else:
                        self.logger.error("A CatalogReference specifies a reference that is not an Entity. Skipping...")

                for vehicle in obj.iter("Vehicle"):
                    self._extract_vehicle_information(obj, rolename, vehicle, args)

                for pedestrian in obj.iter("Pedestrian"):
                    self._extract_pedestrian_information(obj, rolename, pedestrian, args)

                for misc in obj.iter("MiscObject"):
                    self._extract_misc_information(obj, rolename, misc, args)

    def _extract_vehicle_information(self, obj, rolename, vehicle, args):
        """
        Helper function to _set_actor_information for getting vehicle information from XML tree
        """
        color = None
        model = vehicle.attrib.get('name', "vehicle.*")
        category = vehicle.attrib.get('category', "car")
        ego_vehicle = False
        for prop in obj.iter("Property"):
            if prop.get('name', '') == 'type':
                ego_vehicle = prop.get('value') == 'ego_vehicle'
            if prop.get('name', '') == 'color':
                color = prop.get('value')

        speed = self._get_actor_speed(rolename)
        new_actor = ActorConfigurationData(
            model, carla.Transform(), rolename, speed, color=color, category=category, args=args)
        new_actor.transform = self._get_actor_transform(rolename)

        if ego_vehicle:
            self.ego_vehicles.append(new_actor)
        else:
            self.other_actors.append(new_actor)

    def _extract_pedestrian_information(self, obj, rolename, pedestrian, args):
        """
        Helper function to _set_actor_information for getting pedestrian information from XML tree
        """
        model = pedestrian.attrib.get('model', "walker.*")

        new_actor = ActorConfigurationData(model, carla.Transform(), rolename, category="pedestrian", args=args)
        new_actor.transform = self._get_actor_transform(rolename)

        self.other_actors.append(new_actor)

    def _extract_misc_information(self, obj, rolename, misc, args):
        """
        Helper function to _set_actor_information for getting vehicle information from XML tree
        """
        category = misc.attrib.get('category')
        if category == "barrier":
            model = "static.prop.streetbarrier"
        elif category == "guardRail":
            model = "static.prop.chainbarrier"
        else:
            model = misc.attrib.get('name')
        new_actor = ActorConfigurationData(model, carla.Transform(), rolename, category="misc", args=args)
        new_actor.transform = self._get_actor_transform(rolename)

        self.other_actors.append(new_actor)

    def _get_actor_transform(self, actor_name):
        """
        Get the initial actor transform provided by the Init section

        Note: - The OpenScenario specification allows multiple definitions. We use the _first_ one
              - The OpenScenario specification allows different ways of specifying a position.
                We currently support the specification with absolute world coordinates and the relative positions
                RelativeWorld, RelativeObject and RelativeLane
              - When using relative positions the relevant reference point (e.g. transform of another actor)
                should be defined before!
        """

        actor_transform = carla.Transform()

        actor_found = False

        for private_action in self.init.iter("Private"):
            if private_action.attrib.get('object', None) == actor_name:
                if actor_found:
                    # pylint: disable=line-too-long
                    self.logger.warning(
                        "Warning: The actor '%s' was already assigned an initial position. Overwriting pose!", actor_name)
                    # pylint: enable=line-too-long
                actor_found = True
                for position in private_action.iter('Position'):
                    transform = OpenScenarioParser.convert_position_to_transform(
                        position, actor_list=self.other_actors)
                    if transform:
                        actor_transform = transform

        if not actor_found:
            # pylint: disable=line-too-long
            self.logger.warning(
                "Warning: The actor '%s' was not assigned an initial position. Using (0,0,0)", actor_name)
            # pylint: enable=line-too-long

        return actor_transform

    def _get_actor_speed(self, actor_name):
        """
        Get the initial actor speed provided by the Init section
        """
        actor_speed = 0
        actor_found = False

        for private_action in self.init.iter("Private"):
            if private_action.attrib.get('object', None) == actor_name:
                if actor_found:
                    # pylint: disable=line-too-long
                    self.logger.warning(
                        "Warning: The actor '%s' was already assigned an initial position. Overwriting pose!", actor_name)
                    # pylint: enable=line-too-long
                actor_found = True

                for longitudinal_action in private_action.iter('Longitudinal'):
                    for speed in longitudinal_action.iter('Speed'):
                        for target in speed.iter('Target'):
                            for absolute in target.iter('Absolute'):
                                speed = float(absolute.attrib.get('value', 0))
                                if speed >= 0:
                                    actor_speed = speed
                                else:
                                    raise AttributeError(
                                        "Warning: Speed value of actor {} must be positive. Speed set to 0.".format(actor_name))  # pylint: disable=line-too-long
        return actor_speed

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
