"""Interface to the CARLA driving simulator.

This interface must currently be used in `2D compatibility mode`.

This interface has been tested with `CARLA <https://carla.org/>`_ versions 0.9.9,
0.9.10, and 0.9.11.
It supports dynamic scenarios involving vehicles, pedestrians, and props.

The interface implements the :obj:`scenic.domains.driving` abstract domain, so any
object types, behaviors, utility functions, etc. from that domain may be used freely.
For details of additional CARLA-specific functionality, see the world model
:obj:`scenic.simulators.carla.model`.
"""

# Only import CarlaSimulator if the carla package is installed; otherwise the
# import would raise an exception.
carla = None
try:
    import carla
except ImportError:
    pass
if carla:
    from .simulator import CarlaSimulator
del carla
