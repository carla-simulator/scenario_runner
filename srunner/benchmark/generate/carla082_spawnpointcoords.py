#!/usr/bin/env python

# Copyright (c) Adam Gleave
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Save to a JSON file the transformation associated with each spawn point for Carla 0.8.2.

This is intended to be used to translate old configs defined in terms of spawn points to
new config format that uses coordinates."""

import argparse
import json
import logging
import time

from carla.client import make_carla_client
from carla.planner.map import CarlaMap
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError


def save_spawnpoints(args):
    res = []
    with make_carla_client(args.host, args.port) as client:
        print('CarlaClient connected')

        # We load the default settings to the client.
        scene = client.load_settings(CarlaSettings())
        print("Received the start positions")

        # We get the number of player starts, in order to detect the city.
        number_of_player_starts = len(scene.player_start_spots)
        if number_of_player_starts > 100:  # WARNING: unsafe way to check for city, see issue #313
            map = 'Town01'
        else:
            map = 'Town02'

        for spawn_point in scene.player_start_spots:
            location = spawn_point.location
            orientation = spawn_point.orientation
            rotation = spawn_point.rotation
            d = {
                'location': {k: getattr(location, k) for k in ['x', 'y', 'z']},
                'orientation': {k: getattr(orientation, k) for k in ['x', 'y']},
                'rotation': {k: getattr(rotation, k) for k in ['pitch', 'roll', 'yaw']},
            }
            res.append(d)

    with open(args.out, 'w') as f:
        json.dump({map: res}, f)


def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='localhost',
        help='IP of the host server (default: localhost)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-o', '--out',
        metavar='F',
        type=str,
        help='path to save JSON output'
    )

    args = argparser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    while True:
        try:
            save_spawnpoints(args)
            print('Done.')
            return

        except TCPConnectionError as error:
            logging.error(error)
            time.sleep(1)
        except RuntimeError as error:
            logging.error(error)
            break


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
