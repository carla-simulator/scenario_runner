#!/usr/bin/env python

# Copyright (c) 2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides the key configuration parameters for an XML-based scenario
"""

import carla


class ActorConfigurationData(object):

    """
    This is a configuration base class to hold model and transform attributes
    """

    def __init__(self, model, transform, rolename='other', speed=0, autopilot=False,
                 random=False, amount=1, color=None, category="car", args=None):
        self.model = model
        self.rolename = rolename
        self.transform = transform
        self.speed = speed
        self.autopilot = autopilot
        self.random_location = random
        self.amount = amount
        self.color = color
        self.category = category
        self.args = args


class ActorConfiguration(ActorConfigurationData):

    """
    This class provides the basic actor configuration for a
    scenario:
    - Location and rotation (transform)
    - Model (e.g. Lincoln MKZ2017)
    """

    def __init__(self, node, rolename):

        pos_x = float(node.attrib.get('x', 0))
        pos_y = float(node.attrib.get('y', 0))
        pos_z = float(node.attrib.get('z', 0))
        yaw = float(node.attrib.get('yaw', 0))

        random_location = False
        if 'random_location' in node.keys():
            random_location = True

        autopilot = False
        if 'autopilot' in node.keys():
            autopilot = True

        amount = 1
        if 'amount' in node.keys():
            amount = int(node.attrib['amount'])

        super(ActorConfiguration, self).__init__(node.attrib.get('model', 'vehicle.*'),
                                                 carla.Transform(carla.Location(x=pos_x, y=pos_y, z=pos_z),
                                                                 carla.Rotation(yaw=yaw)),
                                                 node.attrib.get('rolename', rolename),
                                                 autopilot, random_location, amount)


class WeatherConfiguration(object):

    """
    This class provides basic weather configuration values
    """

    cloudiness = -1
    precipitation = -1
    precipitation_deposits = -1
    wind_intensity = -1
    sun_azimuth = -1
    sun_altitude = -1
    wetness = -1
    fog_distance = -1
    fog_density = -1


class ScenarioConfiguration(object):

    """
    This class provides a basic scenario configuration incl.:
    - configurations for all actors
    - town, where the scenario should be executed
    - name of the scenario (e.g. ControlLoss_1)
    - type is the class of scenario (e.g. ControlLoss)
    """

    trigger_points = []
    ego_vehicles = []
    other_actors = []
    town = None
    name = None
    type = None
    target = None
    route = None
    agent = None
    weather = WeatherConfiguration()
    friction = None
    subtype = None
