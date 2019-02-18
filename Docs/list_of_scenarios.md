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

###VehicleTurningRight
In this scenario the ego vehicle takes a right turn from an intersection where
a cyclist suddenly drives into the way of the ego vehicle,which has to stop
accordingly. After some time, the cyclist clears the road, such that ego vehicle
can continue driving.

###VehicleTurningLeft
This scenario is similar to 'VehicleTurningRight'. The differnce is that the ego
vehicle takes a left turn from an intersection.

### PassingFromOppositeDirections
Scenario used to calibrating and testing sensor models: ego vehicle passes other vehicle from opposite direction on a long straight.
Available configurations:
PassingFromOppositeDirectionsCar
PassingFromOppositeDirectionsTruck
PassingFromOppositeDirectionsMotorcycle
PassingFromOppositeDirectionsBicycle

### OvertakingSlowTarget
Scenario used for testing a simple overtake maneuver. Ego car should overtake other vehicle with safe margin on a long straight.
Available configurations:
OvertakingSlowTargetCar
OvertakingSlowTargetTruck
OvertakingSlowTargetMotorcycle
OvertakingSlowTargetBicycle

### FollowingAcceleratingTarget
Scenario used for testing whether ego car follows a slowly accelerating other vehicle correctly on a long straight.
Available configurations:
FollowingAcceleratingTargetCar
FollowingAcceleratingTargetTruck
FollowingAcceleratingTargetMotorcycle
FollowingAcceleratingTargetBicycle

### FollowingDeceleratingTarget
Scenario used for testing whether ego car follows a slowly decelerating other vehicle correctly on a long straight.
Available configurations:
FollowingDeceleratingTargetCar
FollowingDeceleratingTargetTruck
FollowingDeceleratingTargetMotorcycle
FollowingDeceleratingTargetBicycle

### FollowingChangingLanesTarget
Scenario used for testing a cut off situation where other vehicle is slowly changing lane to one occupied by ego.
Available configurations:
FollowingChangingLanesCar
FollowingChangingLanesTruck
FollowingChangingLanesMotorcycle
FollowingChangingLanesBicycle

### DrivingOffDriveway
Scenario used for testing a situation where other vehicle is merging from covered driveway right in front of ego.

### OncomingTargetDriftsOntoEgoLane
Scenario where oncoming other vehicle is slowly drifting from its lane onto one occupied by ego. Once ego applies countermeasure(s) to avoid oncoming traffic, it will veer off onto its proper lane.
Available configurations:
OncomingTargetDriftsOntoEgoLaneCar
OncomingTargetDriftsOntoEgoLaneTruck
OncomingTargetDriftsOntoEgoLaneMotorcycle

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

### ControlLoss
In this scenario control loss of a vehicle is tested due to bad road conditions, etc
and it checks whether the vehicle is regained its control and corrected its course.

