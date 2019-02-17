
from Challenge.envs.sensor_interface import SensorInterface

class AutonomousAgent():
    def __init__(self):
        #  current global plans to reach a destination
        self._global_plan = None,

        # this data structure will contain all sensor data
        self.sensor_interface  = SensorInterface()

        # agent's initialization
        self.setup()

    def setup(self):
        """
        Initialize everything needed by your agent.
        """
        pass

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
        sensors = []

        return sensors

    def run_step(self):
        """
        Execute one step of navigation.
        :return: control
        """
        pass

    def __call__(self):
        input_data = self.sensor_interface.get_data()

        control = self.run_step(input_data)
        control.manual_gear_shift = False

        return control

    def all_sensors_ready(self):
        return self.sensor_interface.all_sensors_ready()

    def set_global_plan(self, global_plan):
        self._global_plan = global_plan