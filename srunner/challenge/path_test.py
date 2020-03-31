import glob
import os
import sys

carla_path = '/home/lyq/CARLA_simulator/CARLA_095/PythonAPI'

# original
try:
    sys.path.append(glob.glob(carla_path + '/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

try:

    path = carla_path + '/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64')

    glob_path = glob.glob(path)

    value = glob.glob(carla_path + '/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0]

    sys.path.append(glob.glob(carla_path + '/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

sys.path.append("/home/lyq/CARLA_simulator/CARLA_095/PythonAPI/carla")
sys.path.append("/home/lyq/CARLA_simulator/CARLA_095/PythonAPI/carla/agents")

path = sys.path


try:
    import carla
    location = carla.Location(x=1, y=1, z=1)

except IndexError:
    pass

location = carla.Location(x=1, y=1, z=1)

print("done")

