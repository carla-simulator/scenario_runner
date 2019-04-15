#!/usr/bin/env python

# Copyright (c) Adam Gleave
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Create an XML route configuration file for each (town, pose) config in specified YAML file.

This translates old 0.8.x benchmark configs to new 0.9.x scenario runner route definitions."""

import argparse
import json
import os

import lxml.etree as ET
import yaml


def load_pose_to_coords(pose_to_coords_path):
    with open(pose_to_coords_path, 'r') as f:
        return json.load(f)


def load_yaml_pose(yaml_cfg_path):
    with open(yaml_cfg_path, 'r') as f:
        return yaml.load(f)['poses']


def pose_to_coord(pose_to_coords, town, pose):
    # TODO: need to do more translation!
    d = pose_to_coords[town][pose]
    res = dict(**d['location'], **d['rotation'])
    if town == 'Town01':
        # z coordinate differs.
        # Shift so they have the same mean as current spawn points in Town01 for 0.9.5
        # Town02 does not seem to have changed z between 0.8.x and 0.9.x
        res['z'] -= (39.430625915527344 - 1.3207979733345068)
    res = {k: str(v) for k, v in res.items()}
    return res


def parser():
    argparser = argparse.ArgumentParser(description=__doc__)
    script_dir = os.path.dirname(os.path.realpath(__file__))
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '-p', '--pose-to-coords',
        metavar='F',
        type=str,
        default=os.path.join(script_dir, 'pose_to_coords.json'),
        help='path to save JSON output'
    )
    argparser.add_argument(
        '-e', '--experiment-cfg',
        metavar='F',
        type=str,
        help='path to YAML config file for experiment'
    )
    argparser.add_argument(
        '-o', '--out-dir',
        metavar='D',
        type=str,
        help='directory to save route XML files'
    )

    return argparser.parse_args()


def main():
    args = parser()

    pose_to_coords = load_pose_to_coords(args.pose_to_coords)
    cfg = load_yaml_pose(args.experiment_cfg)
    os.makedirs(args.out_dir, exist_ok=True)

    for town, routes in cfg.items():
        for route_name, route in routes.items():
            root = ET.Element("routes")
            for i, (start_pose, end_pose) in enumerate(route):
                route_xml = ET.SubElement(root, "route", id=str(i), map=town)
                start_coord = pose_to_coord(pose_to_coords, town, start_pose)
                end_coord = pose_to_coord(pose_to_coords, town, end_pose)
                ET.SubElement(route_xml, "waypoint", **start_coord)
                ET.SubElement(route_xml, "waypoint", **end_coord)
            tree = ET.ElementTree(root)
            out_path = os.path.join(args.out_dir, '{}_{}.xml'.format(town, route_name))
            tree.write(out_path, pretty_print=True, xml_declaration=True)


if __name__ == '__main__':
    main()
