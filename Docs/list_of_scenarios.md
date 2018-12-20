# List of Supported Scenarios

Welcome to the ScenarioRunner for CARLA! This document provides a list of all
currently supported scenarios, and a short description for each one.

## Scenarios in Town01
The following scenarios take place in Town01. Hence, the CARLA server has to be
started with Town01, e.g. when not using manual-mode:
```
CarlaUE4.sh /Game/Carla/Maps/Town01 -benchmark -fps=20 -windowed
```

### FollowLeadingVehicle
The scenario realizes a common driving behavior, in which the user-controlled
ego vehicle follows a leading car driving down a given road in Town01. At some
point the leading car slows down and finally stops. The ego vehicle has to react
accordingly to avoid a collision. The scenario ends either via a timeout, or if
the ego vehicle stopped close enough to the leading vehicle

### FollowLeadingVehicleWithObstacle
This scenario is very similar to 'FollowLeadingVehicle'. The only difference is,
that in front of the leading vehicle is a (hidden) obstacle that blocks the way.


## Scenarios in Town03
The following scenarios take place in Town01. Hence, the CARLA server has to be
started with Town03, e.g. when not using manual-mode:
```
CarlaUE4.sh /Game/Carla/Maps/Town03 -benchmark -fps=20 -windowed
```

### OppositeVehicleRunningRedLight
In this scenario an illegal behavior at an intersection is tested. An other
vehicle waits at an intersection, but illegally runs a red traffic light. The
approaching ego vehicle has to handle this situation correctly, i.e. despite of
a green traffic light, it has to stop and wait until the intersection is clear
again. Afterwards, it should continue driving.

### StationaryObjectCrossing
In this scenario a cyclist is stationary waiting in the middle of the road and
blocking the way for the ego vehicle. Hence, the ego vehicle has to stop in
front of the cyclist.

### DynamicObjectCrossing
This is similar to 'StationaryObjectCrossing', but with the difference that the
cyclist is dynamic. It suddenly drives into the way of the ego vehicle, which
has to stop accordingly. After some time, the cyclist will clear the road, such
that the ego vehicle can continue driving.

### NoSignalJunctionCrossing
This scenario tests negotiation between two vehicles crossing cross each other
through a junction without signal.
The ego vehicle is passing through a junction without traffic lights
And encounters another vehicle passing across the junction. The ego vehicle has
to avoid collision and navigate accross the junction to succeed.
