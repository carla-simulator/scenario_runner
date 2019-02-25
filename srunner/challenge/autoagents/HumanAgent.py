import cv2
import numpy as np
import time
from threading import Thread

try:
    import pygame
    from pygame.locals import K_DOWN
    from pygame.locals import K_LEFT
    from pygame.locals import K_RIGHT
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_d
    from pygame.locals import K_s
    from pygame.locals import K_w
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

import carla

from srunner.challenge.autoagents.autonomous_agent import AutonomousAgent

class HumanInterface():
    """
    Class to control a vehicle manually for debugging purposes
    """
    def __init__(self, parent):
        self.quit = False
        self._parent = parent
        self.WIDTH = 800
        self.HEIGHT = 600
        self.THROTTLE_DELTA = 0.05
        self.STEERING_DELTA = 0.01

        pygame.init()
        pygame.font.init()
        self._clock = pygame.time.Clock()
        self._display = pygame.display.set_mode((self.WIDTH, self.HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption("Human Agent")

    def run(self):
        while not self._parent.agent_engaged:
            time.sleep(0.5)

        throttle = 0
        steering = 0
        brake = 0
        while not self.quit:
            self._clock.tick_busy_loop(20)
            # Process events
            keys = pygame.key.get_pressed()
            if keys[K_UP] or keys[K_w]:
                throttle += self.THROTTLE_DELTA
            elif keys[K_DOWN] or keys[K_s]:
                brake += 4*self.THROTTLE_DELTA
                throttle = 0.0
            else:
                throttle -= self.THROTTLE_DELTA / 5.0
                brake = 0.0

            if keys[K_LEFT] or keys[K_a]:
                steering -= self.STEERING_DELTA
            elif keys[K_RIGHT] or keys[K_d]:
                steering += self.STEERING_DELTA
            else:
                steering = 0.0

            pygame.event.pump()

            # normalize values
            steering = min(1.0, max(-1.0, steering))
            throttle = min(1.0, max(0.0, throttle))
            brake = min(1.0, max(0.0, brake))


            self._parent.current_control.steer = steering
            self._parent.current_control.throttle = throttle
            self._parent.current_control.brake = brake



            input_data = self._parent.sensor_interface.get_data()
            image_center = input_data['Center'][1]
            image_left = input_data['Left'][1]
            image_right = input_data['Right'][1]
            image_rear = input_data['Rear'][1]

            top_row = np.hstack((image_left, image_center, image_right))
            bottom_row = np.hstack((0*image_rear, image_rear, 0*image_rear))
            comp_image = np.vstack((top_row, bottom_row))

            # resize image
            image_rescaled = cv2.resize(comp_image, dsize=(self.WIDTH, self.HEIGHT), interpolation=cv2.INTER_CUBIC)

            # display image
            self._surface = pygame.surfarray.make_surface(image_rescaled.swapaxes(0, 1))

            if self._surface is not None:
                self._display.blit(self._surface, (0, 0))

            pygame.display.flip()

        pygame.quit()


class HumanAgent(AutonomousAgent):

    def setup(self):
        self.agent_engaged = False
        self.current_control = carla.VehicleControl()
        self.current_control.steer = 0.0
        self.current_control.throttle = 1.0
        self.current_control.brake = 0.0
        self.current_control.hand_brake = False

        self._hic = HumanInterface(self)
        self._thread = Thread(target=self._hic.run)
        self._thread.start()



    def sensors(self):
        """
        Define the sensor suite required by the agent

        :return: a list containing the required sensors in the following format:

        [
            ['sensor.camera.rgb', {'x':x_rel, 'y': y_rel, 'z': z_rel,
                                   'yaw': yaw, 'pitch': pitch, 'roll': roll,
                                   'width': width, 'height': height, 'fov': fov}, 'Sensor01'],
            ['sensor.camera.rgb', {'x':x_rel, 'y': y_rel, 'z': z_rel,
                                   'yaw': yaw, 'pitch': pitch, 'roll': roll,
                                   'width': width, 'height': height, 'fov': fov}, 'Sensor02'],

            ['sensor.lidar.ray_cast', {'x':x_rel, 'y': y_rel, 'z': z_rel,
                                       'yaw': yaw, 'pitch': pitch, 'roll': roll}, 'Sensor03']
        ]

        """
        sensors = [['sensor.camera.rgb',
                   {'x':0.7, 'y':0.0, 'z':1.60, 'roll':0.0, 'pitch':0.0, 'yaw':0.0, 'width':300, 'height':200,
                    'fov':100},
                   'Center'],

                   ['sensor.camera.rgb',
                    {'x': 0.7, 'y': -0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': -45.0, 'width': 300,
                     'height': 200,
                     'fov': 100},
                    'Left'],

                   ['sensor.camera.rgb',
                    {'x': 0.7, 'y': 0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 45.0, 'width': 300, 'height': 200,
                     'fov': 100},
                    'Right'],

                   ['sensor.camera.rgb',
                    {'x': -1.8, 'y': 0, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 180.0, 'width': 300,
                     'height': 200,
                     'fov': 130},
                    'Rear'],

                    ['sensor.other.gnss', {'x': 0.7, 'y': -0.4, 'z': 1.60},
                     'GPS'],
                   ]


        return sensors

    def run_step(self, input_data):
        self.agent_engaged = True

        # print("=== steering = {}, throttle = {}, brake = {}".format(self.current_control.steer,
        #                                                       self.current_control.throttle,
        #                                                             self.current_control.brake))
        return self.current_control

    def destroy(self):
        self._hic.quit = True
        self._thread.join()
