
from srunner.challenge.envs.sensor_interface import SensorInterface

class AutonomousAgent():
    def __init__(self, path_to_conf_file):
        #  current global plans to reach a destination
        self._global_plan = None,

        # this data structure will contain all sensor data
        self.sensor_interface  = SensorInterface()

        # agent's initialization
        self.setup(path_to_conf_file)

    def setup(self, path_to_conf_file):
        """
        Initialize everything needed by your agent.
        """
        pass

    def sensors(self):
        """
        Define the sensor suite required by the agent

        :return: a list containing the required sensors in the following format:

        [
            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 300, 'height': 200, 'fov': 100, 'id': 'Left'},

            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': 0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 300, 'height': 200, 'fov': 100, 'id': 'Right'},

            {'type': 'sensor.lidar.ray_cast', 'x': 0.7, 'y': 0.0, 'z': 1.60, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0,
             'id': 'LIDAR'}
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

    def destroy(self):
        """
        Destroy (clean-up) the agent
        :return:
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