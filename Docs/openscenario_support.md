## OpenSCENARIO Support

The scenario_runner provides support for the [OpenSCENARIO](http://www.openscenario.org/) 1.0 standard.
The current implementation covers initial support for maneuver Actions, Conditions, Stories and the Storyboard.
If you would like to use evaluation criteria for a scenario to evaluate pass/fail results, these can be implemented
as StopTriggers (see below). However, not all features for these elements are yet available. If in doubt, please see the
module documentation in srunner/tools/openscenario_parser.py

An example for a supported scenario based on OpenSCENARIO is available [here](https://github.com/carla-simulator/scenario_runner/blob/master/srunner/examples/FollowLeadingVehicle.xosc)

In addition, it is recommended to take a look into the official documentation available [here](https://releases.asam.net/OpenSCENARIO/1.0.0/Model-Documentation/index.html) and [here](https://releases.asam.net/OpenSCENARIO/1.0.0/ASAM_OpenSCENARIO_BS-1-2_User-Guide_V1-0-0.html#_foreword).

### Migrating OpenSCENARIO 0.9.x to 1.0
The easiest way to convert old OpenSCENARIO samples to the official standard 1.0 is to use _xsltproc_ and the migration scheme located in the openscenario folder.
Example:

```bash
xsltproc -o newScenario.xosc migration0_9_1to1_0.xslt oldScenario.xosc
```


### Level of support
In the following the OpenSCENARIO attributes are listed with their current support status.

#### General OpenSCENARIO setup

This covers all part that are defined outside the OpenSCENARIO Storyboard

| Attribute                          | Support                            | Notes                              |
| ---------------------------------------- | ---------------------------------------- | ---------------------------------------- |
| `FileHeader`                       | &#9989;                              | Use "CARLA:" at the beginning of the description to use the CARLA coordinate system.                                |
| `CatalogLocations`<br>`ControllerCatalog`                                      | &#9989;                              | While the catalog is supported, the reference/usage may not work.                |
| `CatalogLocations`<br>`EnvironmentCatalog`                                     | &#9989;                              |                                    |
| `CatalogLocations`<br>`ManeuverCatalog`                                        | &#9989;                              |                                    |
| `CatalogLocations`<br>`MiscObjectCatalog`                                      | &#9989;                              |                                    |
| `CatalogLocations`<br>`PedestrianCatalog`                                      | &#9989;                              |                                    |
| `CatalogLocations`<br>`RouteCatalog`                                           | &#9989;                              | While the catalog is supported, the reference/usage may not work.                |
| `CatalogLocations`<br>`TrajectoryCatalog`                                      | &#9989;                              | While the catalog is supported, the reference/usage may not work.                |
| `CatalogLocations`<br>`VehicleCatalog`                                         | &#9989;                              |                                    |
| `ParameterDeclarations`            | &#9989;                              |                                    |
| `RoadNetwork`<br>`LogicFile`                                                   | &#9989;                              | The CARLA level can be used directly (e.g. LogicFile=Town01). Also any OpenDRIVE path can be provided.              |
| `RoadNetwork`<br>`SceneGraphFile`                                              | &#10060;                               | The provided information is not used.                                            |
| `RoadNetwork`<br>`TafficSignals`                                               | &#10060;                               | The provided information is not used.                                            |
| `Entities`<br>`EntitySelection`                                                | &#10060;                               | The provided information is not used.                                            |
| `Entities` `ScenarioObject`<br>`CatalogReference`                               | &#9989;                              | The provided information is not used.                                            |
| `Entities` `ScenarioObject`<br>`MiscObject`                                     | &#9989;                              | The name should match a CARLA vehicle model, otherwise a default vehicle based on the vehicleCategory is used. BoundingBox entries are ignored.                   |
| `Entities` `ScenarioObject`<br>`ObjectController`                               | &#10060;                               | The provided information is not used.                                            |
| `Entities` `ScenarioObject`<br>`Pedestrian`                                     | &#9989;                              | The name should match a CARLA vehicle model, otherwise a default vehicle based on the vehicleCategory is used. BoundingBox entries are ignored.                   |
| `Entities` `ScenarioObject`<br>`Vehicle`                                        | &#9989;                              | The name should match a CARLA vehicle model, otherwise a default vehicle based on the vehicleCategory is used. The color can be set via properties ('Property name="color" value="0,0,255"'). Axles, Performance, BoundingBox entries are ignored. |

<br>


#### OpenSCENARIO Storyboard

##### OpenSCENARIO Actions

The OpenSCENARIO Actions can be used for two different purposes. First, Actions can be used to
define the initial behavior of something, e.g. a traffic participant. Therefore, Actions can be
used within the OpenSCENARIO Init. In addition, Actions are also used within the OpenSCENARIO
story. In the following, the support status for both application areas is listed. If an action
contains of submodules, which are not listed, the support status applies to all submodules.

###### GlobalAction



| GlobalAction                          | Init  support                            | Story support                              | Notes                              |
| ---------------------------------------- | ---------------------------------------- | ---------------------------------------- | ---------------------------------------- |
| `EntityAction`                | &#9989;                          | &#9989;                          |                               |
| `EnvironmentAction`           | &#9989;                         | &#10060;                          |                               |
| `ParameterAction`             | &#9989;                          | &#9989;                          | Values may be literals, `$name` references or `${...}` expressions (see [Parameters, variables and expressions](#parameters-variables-and-expressions)). The parameter must be declared. |
| `VariableAction`              | &#9989;                          | &#9989;                          | `SetAction` and `ModifyAction` (`AddValue`, `MultiplyByValue`). See [Parameters, variables and expressions](#parameters-variables-and-expressions). |
| `InfrastructureAction` `TrafficSignalAction`<br>`TrafficAction`                                               | &#10060;                          | &#10060;                          |                               |
| `InfrastructureAction` `TrafficSignalAction`<br>`TrafficSignalControllerAction`                               | &#10060;                          | &#9989;                          |                               |
| `InfrastructureAction` `TrafficSignalAction`<br>`TrafficSignalStateAction`                                    | &#10060;                          | &#9989;                         | As traffic signals in CARLA towns have no unique ID, they have to be set by providing their position (Example: TrafficSignalStateAction name="pos=x,y" state="green"). The id can also be used for none CARLA town (Example: TrafficSignalStateAction name="id=n" state="green") |

<br>



###### UserDefinedAction



| UserDefinedAction                          | Init  support                            | Story support                              | Notes                              |
| ---------------------------------------- | ---------------------------------------- | ---------------------------------------- | ---------------------------------------- |
| `CustomCommandAction`                       | &#10060;                                        | &#9989;                                       | This action is currently used to trigger the execution of an additional script. Example: type="python /path/to/script args". |

<br>


###### PrivateAction


| PrivateAction                          | Init  support                            | Story support                              | Notes                              |
| ---------------------------------------- | ---------------------------------------- | ---------------------------------------- | ---------------------------------------- |
| `ActivateControllerAction`                          | &#10060;             | &#9989;            | Can be used to activate/deactive the CARLA autopilot.                                             |
| `ControllerAction`                                  | &#9989;            | &#9989;            | AssignControllerAction is supported, but a Python module has to be provided for the controller implementation, and in OverrideControllerValueAction all values need to be `False`. |
| `LateralAction`<br> `LaneChangeAction`             | &#10060;             | &#9989;            | Currently all lane changes have a linear dynamicShape, the dynamicDimension is defined as the distance and are relative to the actor itself (RelativeTargetLane).                  |
| `LateralAction`<br>`LaneOffsetAction`             | &#10060;             | &#9989;             |  Currently all type of dynamicShapes are ignored and depend on the controller. This action might not work as intended if the offset is high enough to make the vehicle exit its lane  |
| `LateralAction`<br> `LateralDistanceAction`        | &#9989;             | &#9989;             |                  |
| `LongitudinalAction`<br> `LongitudinalDistanceAction`| &#10060;             | &#9989;              |`timeGap` attribute is not supported              |
| `LongitudinalAction`<br> `SpeedAction`             | &#9989;            | &#9989;            |                  |
| `SynchronizeAction`                                 | &#10060;             | &#10060;             |                  |
| `TeleportAction`                                    | &#9989;            | &#9989;            |                  |
| `VisibilityAction`                                  | &#10060;             | &#10060;             |                  |
| `RoutingAction`<br> `AcquirePositionAction`        | &#9989;             | &#9989;            | The target position is reached with the `fastest` route option. See [Routes, trajectories and speed profiles](#routes-trajectories-and-speed-profiles). |
| `RoutingAction`<br> `AssignRouteAction`            | &#9989;             | &#9989;            | Route Options (shortest/fastest/etc) are supported. Shortests means direct path between A and B, all other will use the shortest path along the road network between A and B. Inline routes and `CatalogReference`s are supported. |
| `RoutingAction`<br> `FollowTrajectoryAction`       | &#9989;             | &#9989;             | Only `Polyline` shapes. `TimeReference` (`None` or `Timing` with offset/scale, absolute or relative) is applied to the vertex times; the `trajectoryFollowingMode` is kept but executed by the actor's controller. `TrajectoryRef` (OpenSCENARIO 1.1+) is supported. |
| `LongitudinalAction`<br> `SpeedProfileAction`      | &#9989;             | &#9989;             | OpenSCENARIO 1.2+. Each entry becomes a target speed step; see [Routes, trajectories and speed profiles](#routes-trajectories-and-speed-profiles). |

<br>



##### OpenSCENARIO Conditions

Conditions in OpenSCENARIO can be defined either as ByEntityCondition or as ByValueCondition.
Both can be used for StartTrigger and StopTrigger conditions.
The following two tables list the support status for each.

###### ByEntityCondition


| EntityCondition                                | Support                                        | Notes                                          |
| ---------------------------------------------- | ---------------------------------------------- | ---------------------------------------------- |
| `AccelerationCondition`                        | &#9989;                                          |                                                |
| `CollisionCondition`                           | &#9989;                                          |                                                |
| `DistanceCondition`                            | &#9989;                                          | \*freespace\* attribute is still not supported |
| `EndOfRoadCondition`                           | &#9989;                                          |                                                |
| `OffroadCondition`                             | &#9989;                                          |                                                |
| `ReachPositionCondition`                       | &#9989;                                          |                                                |
| `RelativeAngleCondition`                       | &#9989;                                          | OpenSCENARIO 1.2+. `coordinateSystem` `entity` (default) and `world` only |
| `RelativeClearanceCondition`                   | &#9989;                                          | OpenSCENARIO 1.2+. See [Routes, trajectories and speed profiles](#routes-trajectories-and-speed-profiles) |
| `RelativeDistanceCondition`                    | &#9989;                                          |                                                |
| `RelativeSpeedCondition`                       | &#9989;                                          |                                                |
| `SpeedCondition`                               | &#9989;                                          |                                                |
| `StandStillCondition`                          | &#9989;                                          |                                                |
| `TimeHeadwayCondition`                         | &#9989;                                          |                                                |
| `TimeToCollisionCondition`                     | &#9989;                                          |                                                |
| `TraveledDistanceCondition`                    | &#9989;                                          |                                                |

<br>

###### ByValueCondition


| ValueCondition            | Support                   | Notes                     |
| -------------------------------------------------------- | -------------------------------------------------------- | -------------------------------------------------------- |
| `ParameterCondition`         | &#9989;                     | The level of support depends on the parameter. It is recommended to use other conditions if possible. Please also consider the note below. The `value` may be a `$name` reference or a `${...}` expression.                                                  |
| `VariableCondition`          | &#9989;                     | Compares a declared variable with `value` (literal, `$name` parameter reference or `${...}` expression). For `string` variables only `equalTo` and `notEqualTo` are allowed. |
| `SimulationTimeCondition`                                    | &#9989;                     |                           |
| `StoryboardElementStateCondition`                            | &#9989;                     | startTransition, stopTransition, endTransition and completeState are currently supported.                  |
| `TimeOfDayCondition`         | &#9989;                     |                           |
| `TrafficSignalCondition`                                     | &#9989;                     | As traffic signals in CARLA towns have no unique ID, they have to be set by providing their position (Example: TrafficSignalCondition name="pos=x,y" state="green"). The id can also be used for none CARLA town (Example: TrafficSignalCondition name="id=n" state="green") |
| `TrafficSignalControllerCondition`                           | &#9989;                      |                           |
| `UserDefinedValueCondition`                                  | &#10060;                      |                           |

<br>

!!! Note
     In the OpenSCENARIO 1.0 standard, a definition of test / evaluation criteria is not
     defined. For this purpose, you can re-use StopTrigger conditions with CARLA. The following
     StopTrigger conditions for evaluation criteria are supported through ParameterConditions by
     providing the criteria name for the condition:

     * criteria_RunningStopTest
     * criteria_RunningRedLightTest
     * criteria_WrongLaneTest
     * criteria_OnSideWalkTest
     * criteria_KeepLaneTest
     * criteria_CollisionTest
     * criteria_DrivenDistanceTest


#### Parameters, variables and expressions

ParameterDeclarations, VariableDeclarations, `ParameterAction`, `VariableAction`, `ParameterCondition` and
`VariableCondition` are evaluated by a typed semantic layer (`srunner/openscenario/`) when the scenario is built
and while it runs:

* Values can be literals, `$name` parameter references or `${...}` expressions as defined by OpenSCENARIO 1.1+.
  Expressions support numeric and Boolean literals (`true`, `false`, `0`, `1`), parameter references, the operators
  `+`, `-`, `*`, `/`, `%` (remainder), `not`, `and`, `or`, parentheses and the functions `round`, `floor`, `ceil`,
  `sqrt`, `pow`, `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `sign`, `abs`, `max`, `min`. Anything else (for example
  `**`, comparisons, strings or other function names) is rejected. Expressions are parsed safely and never executed
  as Python code.
* Results are converted to the declared type of the target parameter or variable. Integers convert implicitly to
  `double`; a `double` result is only accepted for an integer type when converted explicitly with `round`, `floor`
  or `ceil`. `round` rounds halves away from zero. Division by zero, square roots of negative numbers, NaN, infinity
  and integer overflow are errors.
* ParameterDeclaration and VariableDeclaration values may themselves be references or expressions; they are
  evaluated in declaration order after `--openscenarioparams` overrides have been applied.
* Variables are declared in `VariableDeclarations` and can be changed with `VariableAction` in the `Init` and in
  stories. As required by the standard, variables can only be read through `VariableCondition`; referencing a
  variable from an expression or attribute is an error.
* Parameter changes made by `ParameterAction` are written to the global parameter table read by `$name`
  references, so behaviors that read the parameter after the change see the new value.
* `$name` references and `${...}` expressions are also resolved in the other attributes ScenarioRunner reads, for
  example weather, positions, condition values and `delay`, Boolean flags such as `freespace` or `continuous`,
  entity models (`Vehicle name="$Model"`), properties and catalog paths. Attributes read while the configuration is
  loaded are evaluated with the declaration values and the `--openscenarioparams` overrides. Attributes that name
  an element (`ScenarioObject`, story element and condition names, `storyboardElementRef`) are identifiers and are
  used as written.
* When the configuration is loaded, and again when the scenario is built (before its non-ego actors are spawned),
  the document is checked for duplicate parameter or variable declarations, references to undeclared parameters or
  variables in any attribute, invalid expression syntax and declaration values that do not match their type. Any
  such error aborts the scenario with a `ValidationError` listing all problems. A reference counts as declared if
  the parameter is declared in any `ParameterDeclarations` of the document, because parameter scoping is not
  modelled yet.
* A document that is not an OpenSCENARIO document (wrong root element, missing `FileHeader` or `revMajor`/`revMinor`,
  XML syntax error) raises a `ValidationError` as well. So does an attribute whose value cannot be evaluated while
  the scenario is built (for example a division by zero or a value of the wrong type); each diagnostic names the
  failing attribute value.
* An error that can only be detected while the scenario runs (for example a `VariableAction` value that does not
  match the variable type) makes the corresponding behavior fail and is logged; it does not crash the runner.

Not supported yet:

* `scenario_runner.py --openscenario` still only accepts files declaring OpenSCENARIO 1.0 and validates them against
  the 1.0 schema. Files that contain `VariableDeclarations`, `VariableAction` or `VariableCondition` are therefore
  rejected before they reach the semantic layer; these features are currently only usable through the Python API.
  The 1.0 schema also restricts numeric attributes (for example `AddValue`/`MultiplyByValue` values) to literals and
  `$name` references, so from the command line `${...}` is only accepted in string-typed attributes such as
  `ParameterDeclaration`, `SetAction` and `ParameterCondition` values.
* Parameter scoping (catalog parameters, nested `ParameterDeclarations`), `ConstraintGroup`s, parameter value
  distributions, `MonitorDeclarations`, the rule that `int`, `unsignedInt` and `unsignedShort` parameters must not
  be mixed in one expression, and reading or writing variables from outside the scenario.

#### Routes, trajectories and speed profiles

`AssignRouteAction`, `AcquirePositionAction`, `FollowTrajectoryAction` and `SpeedProfileAction` (in the `Init` and in
stories), and the `RelativeAngleCondition` and `RelativeClearanceCondition` entity conditions, are parsed into typed
objects by the semantic layer and turned into behaviors from those objects:

* Routes and trajectories keep their positions as typed positions until the behavior starts, so relative positions
  are resolved against the actors' poses at that moment.
* `FollowTrajectoryAction` reads `TimeReference` and `TrajectoryFollowingMode` from their child elements. With
  `TimeReference/None` the vertex times are ignored. With `Timing` each vertex time `t` becomes `t * scale + offset`,
  measured from the start of the action (`relative`) or of the simulation (`absolute`); every vertex then needs a
  time. The following mode is passed on, but how closely the actor tracks the trajectory depends on its controller;
  `position` is reported as an approximation warning.
* `SpeedProfileAction` commands each `SpeedProfileEntry` speed as a target speed of the actor's controller. An entry
  with a `time` holds its target for that time before the next entry starts; an entry without a `time`, and the last
  entry, last until the speed is reached. With `entityRef` the speeds are relative to the speed of that entity.
  `DynamicConstraints` and the difference between the `follow` and `position` modes are not enforced (approximation
  warning).
* `RelativeAngleCondition` compares the heading, pitch or roll of the triggering entity relative to the referenced
  entity with `angle` ± `angleTolerance`. The heading follows the OpenSCENARIO convention unless the scenario uses
  CARLA coordinates. The `lane`, `road` and `trajectory` coordinate systems are not supported.
* `RelativeClearanceCondition` is true while no entity (or none of the listed `EntityRef`s) is between
  `distanceBackward` behind and `distanceForward` ahead of the triggering entity, along its heading, in the
  `RelativeLaneRange`s (positive lanes are to its left; no range means all lanes). Missing distances count as 0 m.
  Lanes of the opposite driving direction are only considered with `oppositeLanes="true"`, and `freeSpace="true"`
  measures between bounding boxes. Relative lanes are exact for entities on the same road section and estimated
  from the lateral offset elsewhere.
* Attribute values may be `$name` references or `${...}` expressions; they are evaluated when the scenario is built.

The scenario is aborted with a structured diagnostic instead of running something different from what was written:

* Structural errors (for example a `Polyline` with fewer than two vertices, a `Route` with fewer than two waypoints, a
  missing `TimeReference`, an unknown `followingMode`, a `ParameterAction` or `VariableAction` whose `ModifyAction`
  has no valid `Rule`) raise a `ValidationError`.
* Features the CARLA backend cannot execute raise an `UnsupportedFeatureError` whose diagnostic category is
  `backend_capability_error`: `Clothoid`, `ClothoidSpline` and `Nurbs` trajectories, closed trajectories, a non-zero
  `initialDistanceOffset`, `RoutePosition` and other position types not listed in the table below, and the
  unsupported `RelativeAngleCondition` coordinate systems.
* Entity conditions that are not supported at all raise an `UnsupportedFeatureError` instead of being silently
  dropped.

Other actions keep their previous implementation. As noted above, `scenario_runner.py --openscenario` still only
accepts OpenSCENARIO 1.0 files validated against the 1.0 schema, so `SpeedProfileAction`, `TrajectoryRef`,
`RelativeAngleCondition` and `RelativeClearanceCondition` are currently only usable through the Python API.

##### OpenSCENARIO Positions

There are several ways of defining positions in OpenSCENARIO. In the following we list the
current support status for each definition format.

| Position                          | Support                            | Notes                              |
| ---------------------------------------- | ---------------------------------------- | ---------------------------------------- |
| `LanePosition`           | &#9989;                    |                          |
| `RelativeLanePosition`   | &#9989;                    |                          |
| `RelativeObjectPosition` | &#9989;                    |                          |
| `RelativeRoadPosition`   | &#9989;                     |                          |
| `RelativeWorldPosition`  | &#9989;                    |                          |
| `RoadPosition`           | &#9989;                     |                          |
| `RoutePosition`          | &#10060;                     | Parsed and validated; using it in a routing action aborts the scenario with an unsupported-feature diagnostic |
| `WorldPosition`          | &#9989;                    |                          |

<br>
