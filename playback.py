
import carla

import argparse
import logging

def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='Print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')

    args = argparser.parse_args()


    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(2.0)
        client.replay_file("test3.rec", 0, 0, 0)
    except Exception as error:
        logging.exception(error)



if __name__ == '__main__':
    main()
