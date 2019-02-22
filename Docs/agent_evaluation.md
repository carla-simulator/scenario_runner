#### Setting up your agent for evaluation.

To have your agent evaluated by the challenge evaluation system 
you must define an Agent class that inherit the 
[AutonomousAgent](../srunner/challenge/agents/autonomous_agent.py) base class.

On your agent class there are two main functions to be overwritten
that need to be defined in order to set your agent to run.
Further you also should consider the route to the goal that is
initially set as a variable.



##### The "setup" function:

This function is where you set all the sensors required by your agent.
For instance, on the [dummy agent sample class](../srunner/challenge/agents/DummyAgent.py) the following sensors are defined:

```
    def sensors(self):
        
        sensors = [['sensor.camera.rgb',
           {'x':0.7, 'y':0.0, 'z':1.60, 'roll':0.0, 'pitch':0.0, 'yaw':0.0, 'width':800, 'height':600, 'fov':100},
           'Center'],

           ['sensor.camera.rgb',
            {'x':0.7, 'y':-0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': -45.0, 'width': 800, 'height': 600,
             'fov': 100},
            'Left'],

           ['sensor.camera.rgb',
            {'x':0.7, 'y':0.4, 'z':1.60, 'roll':0.0, 'pitch':0.0, 'yaw':45.0, 'width':800, 'height':600,
             'fov':100},
            'Right'],

           ['sensor.lidar.ray_cast',
            {'x': 0.7, 'y': -0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0,
             'yaw': -45.0},
            'LIDAR'],

            ['sensor.other.gnss', {'x': 0.7, 'y': -0.4, 'z': 1.60},
             'GPS'],
           ]
        return sensors
```

Every sensor is a list of three positions.
The first position specifies the type of sensor. 
The second position specifies the sensor parameters
 such as location and orientation with respect to the vehicle.
 The third position is the label given to  the sensor.



##### The "run_step" function:

This function is called on every step of the simulation from the challenge evaluation
an receives some input data as parameter.

This input data is a dictionary with all the sensors specified on the "setup" function.

This function should return a [vehicle control](https://carla.readthedocs.io/en/latest/python_api_tutorial/#vehicles)
 to be applied into the ego vehicle.




##### The initial route:

On the beginning of the execution, the entire route that the hero agent
should travel is set on  the "self.global_plan" variable:

```
[({'z': 0.0, 'lat': 48.99822669411668, 'lon': 8.002271601998707}, <RoadOption.LANEFOLLOW: 4>), 
 ({'z': 0.0, 'lat': 48.99822669411668, 'lon': 8.002709765148996}, <RoadOption.LANEFOLLOW: 4>),
 ... 
 ({'z': 0.0, 'lat': 48.99822679980298, 'lon': 8.002735250105061}, <RoadOption.LANEFOLLOW: 4>)]`
 ```
 
 It is represented as a list of tuples, containing the route waypoints, expressed in latitude
 and longitude and the current road option recommended. For an intersection the option can
 be go straight, turn left or turn right. For the rest of the route the recommended option
 is lane follow.