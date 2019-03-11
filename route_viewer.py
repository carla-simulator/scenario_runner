import argparse
import xml.etree.ElementTree as ET
import matplotlib as mpl
from random import shuffle
mpl.use('Agg')
import matplotlib.pyplot as plt

import carla

LIFETIME = 600
LSIZE = 64

class RouteViewer():
    def __init__(self, args):
        self.host = args.host
        self.port = args.port
        self.filename = args.route_file
        self.tree = ET.parse(self.filename)

        self.cmap = 255.0 * plt.get_cmap('Pastel1', lut=LSIZE).colors
        self.indexer = [x for x in range(LSIZE)]

        shuffle(self.indexer)

        self.list_routes = []
        # get root element
        root = self.tree.getroot()
        for node in root:
            current_route = []
            route_id = int(node.get("id"))
            route_map = node.get("map")

            for wp in node.iter('waypoint'):
                loc = carla.Location()
                loc.x = float(wp.get("x"))
                loc.y = float(wp.get("y"))
                loc.z = float(wp.get("z"))

                rot = carla.Rotation()
                rot.pitch = float(wp.get("pitch"))
                rot.roll = float(wp.get("roll"))
                rot.yaw = float(wp.get("roll"))
                current_route.append(carla.Transform(loc, rot))

            self.list_routes.append({'id': route_id, 'map': route_map, 'locations': current_route})

    def run(self):
        # let's place the spectator actor in a top view
        client = carla.Client(self.host, self.port)
        client.set_timeout(40)
        world = None
        for route in self.list_routes:
            route_id = route['id']
            map_name = route['map']

            if not world or world.get_map().name != map_name:
                world = client.load_world(map_name)

            spectator = world.get_spectator()

            color = self.get_color(route_id)

            loc = carla.Location(z=400)
            rot = carla.Rotation(pitch=-90)
            spectator.set_transform(carla.Transform(loc, rot))

            trajectory = route['locations']
            start = trajectory[0]
            for idx in range(1, len(trajectory)):
                current = trajectory[idx]


                world.debug.draw_line(start.location, current.location,
                                      thickness=0.8,
                                      color=color,
                                      life_time=LIFETIME)
                # update
                start = current

            world.debug.draw_point(start.location, size=0.1, color=carla.Color(0, 0, 255), life_time=LIFETIME)
            world.debug.draw_point(current.location, size=0.1, color=carla.Color(255, 0, 0), life_time=LIFETIME)
            world.debug.draw_string(start.location,
                                    "[R{}]".format(route_id),
                                    draw_shadow=False,
                                    color=color,
                                    life_time=LIFETIME)

    def get_color(self, idx):
        idx = idx % LSIZE
        color = self.cmap[self.indexer[idx]]

        return carla.Color(int(color[0]), int(color[1]), int(color[2]))



def main():
    argparser = argparse.ArgumentParser(description='Route Viewer')
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
    argparser.add_argument(
        '--route-file',
        type=str,
        default="routes.xml",
        help='Filename for the routes')

    args = argparser.parse_args()

    route_viewer = RouteViewer(args)
    route_viewer.run()

if __name__ == '__main__':
    main()