#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Spawning of actors.

The world, the client, the traffic manager port and the shared random state all
come from SimulationData, which is where the simulator connection lives.
"""

import carla

from srunner.scenariomanager.simulation_data import SimulationData

# Legacy (0.9.x) blueprint id -> its CARLA UE5 equivalent. Two-wheelers keep
# their original ids, so they need no entry.

UE5_BLUEPRINT_ALIASES = {
    'vehicle.lincoln.mkz_2017': 'vehicle.lincoln.mkz',
    'vehicle.lincoln.mkz2017': 'vehicle.lincoln.mkz',
    'vehicle.lincoln.mkz_2020': 'vehicle.lincoln.mkz',
    'vehicle.tesla.model3': 'vehicle.lincoln.mkz',
    'vehicle.audi.tt': 'vehicle.ue4.audi.tt',
    'vehicle.audi.a2': 'vehicle.ue4.audi.tt',
    'vehicle.audi.etron': 'vehicle.ue4.audi.tt',
    'vehicle.bmw.grantourer': 'vehicle.ue4.bmw.grantourer',
    'vehicle.chevrolet.impala': 'vehicle.ue4.chevrolet.impala',
    'vehicle.ford.crown': 'vehicle.ue4.ford.crown',
    'vehicle.ford.mustang': 'vehicle.ue4.ford.mustang',
    'vehicle.mustang.mustang': 'vehicle.ue4.ford.mustang',
    'vehicle.mercedes.coupe': 'vehicle.ue4.mercedes.ccc',
    'vehicle.mercedes.coupe_2020': 'vehicle.ue4.mercedes.ccc',
    'vehicle.mercedes-benz.coupe': 'vehicle.ue4.mercedes.ccc',
    'vehicle.dodge.charger_2020': 'vehicle.dodge.charger',
    'vehicle.dodge.charger_police': 'vehicle.dodgecop.charger',
    'vehicle.dodge.charger_police_2020': 'vehicle.dodgecop.charger',
    'vehicle.dodge_charger.police': 'vehicle.dodgecop.charger',
    'vehicle.carlamotors.carlacola': 'vehicle.carlacola.actors',
    'vehicle.carlamotors.firetruck': 'vehicle.firetruck.actors',
    'vehicle.ford.ambulance': 'vehicle.ambulance.ford',
    'vehicle.volkswagen.t2': 'vehicle.sprinter.mercedes',
    'vehicle.volkswagen.t2_2021': 'vehicle.sprinter.mercedes',
    'vehicle.mercedes.sprinter': 'vehicle.sprinter.mercedes',
    'vehicle.jeep.wrangler_rubicon': 'vehicle.nissan.patrol',
    'vehicle.nissan.patrol_2021': 'vehicle.nissan.patrol',
    'vehicle.mini.cooper_s': 'vehicle.mini.cooper',
    'vehicle.mini.cooper_s_2021': 'vehicle.mini.cooper',
    'vehicle.mini.cooperst': 'vehicle.mini.cooper',
    'vehicle.nissan.micra': 'vehicle.mini.cooper',
    'vehicle.citroen.c3': 'vehicle.mini.cooper',
    'vehicle.seat.leon': 'vehicle.mini.cooper',
    'vehicle.toyota.prius': 'vehicle.lincoln.mkz',
    'static.prop.bench03': 'static.prop.bench02',
    'static.prop.box01': 'static.prop.creasedbox01',
    'static.prop.box02': 'static.prop.creasedbox01',
    'static.prop.box03': 'static.prop.creasedbox01',
    'static.prop.creasedbox02': 'static.prop.creasedbox01',
    'static.prop.creasedbox03': 'static.prop.creasedbox01',
    'static.prop.busstoplb': 'static.prop.busstop',
    'static.prop.clothcontainer': 'static.prop.recyclecardboard',
    'static.prop.glasscontainer': 'static.prop.recycleglass',
    'static.prop.foodcart': 'static.prop.kiosk_01',
    'static.prop.haybalelb': 'static.prop.haybale',
    'static.prop.ironplank': 'static.prop.streetbarrier',
    'static.prop.plantpot08': 'static.prop.plantpot07',
    'static.prop.table': 'static.prop.plastictable',
}

# Pedestrian numbering starts at 0015 in the UE5 content: remap the removed ids.
for _i in range(1, 15):
    UE5_BLUEPRINT_ALIASES['walker.pedestrian.%04d' % _i] = 'walker.pedestrian.%04d' % (_i + 14)
del _i


def _wait_for_actors(tick):
    """Advance the world so that the actors just asked for exist"""
    if tick:
        SimulationData.tick_world()


def resolve_model_name(model):
    # type: (str) -> str
    """
    Translate a legacy (CARLA 0.9.x) blueprint id into its UE5 equivalent, if
    the requested one does not exist in the connected server's library.
    """
    if '*' in model:
        return model
    try:
        SimulationData.get_world().get_blueprint_library().find(model)
        return model  # exists as-is
    except (IndexError, RuntimeError):
        pass
    alias = UE5_BLUEPRINT_ALIASES.get(model)
    if alias is not None:
        print("WARNING: Actor model {} does not exist in this CARLA version. Using {} instead".format(
            model, alias))
        return alias
    return model




def create_blueprint(model, rolename='scenario', color=None, actor_category="car", attribute_filter=None):
    # type: (str, str, carla.Color | None, str, dict | None) -> carla.ActorBlueprint
    """
    Function to setup the blueprint of an actor given its model and other relevant parameters
    """

    def check_attribute_value(blueprint, name, value):
        """
        Checks if the blueprint has that attribute with that value
        """
        if not blueprint.has_attribute(name):
            return False

        attribute_type = blueprint.get_attribute(name).type
        if attribute_type == carla.ActorAttributeType.Bool:
            return blueprint.get_attribute(name).as_bool() == value
        elif attribute_type == carla.ActorAttributeType.Int:
            return blueprint.get_attribute(name).as_int() == value
        elif attribute_type == carla.ActorAttributeType.Float:
            return blueprint.get_attribute(name).as_float() == value
        elif attribute_type == carla.ActorAttributeType.String:
            return blueprint.get_attribute(name).as_str() == value

        return False

    rng = SimulationData.get_rng()

    # Set the model
    try:
        model = resolve_model_name(model)
        blueprints = SimulationData.get_world().get_blueprint_library().filter(model)
        if attribute_filter is not None:
            for key, value in attribute_filter.items():
                blueprints = [x for x in blueprints if check_attribute_value(x, key, value)]

        blueprint = rng.choice(blueprints)
    except ValueError:
        raise ValueError("Actor model {} not available".format(model))

    # Set the color
    if color:
        if not blueprint.has_attribute('color'):
            print(
                "WARNING: Cannot set Color ({}) for actor {} due to missing blueprint attribute".format(
                    color, blueprint.id
                )
            )
        else:
            default_color_rgba = blueprint.get_attribute('color').as_color()
            default_color = '({}, {}, {})'.format(default_color_rgba.r, default_color_rgba.g, default_color_rgba.b)
            try:
                blueprint.set_attribute('color', color)
            except ValueError:
                # Color can't be set for this vehicle
                print(
                    "WARNING: Color ({}) cannot be set for actor {}. Using instead: ({})".format(
                        color, blueprint.id, default_color
                    )
                )
                blueprint.set_attribute('color', default_color)
    else:
        if blueprint.has_attribute('color') and rolename != 'hero':
            color = rng.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)

    # Make pedestrians mortal
    if blueprint.has_attribute('is_invincible'):
        blueprint.set_attribute('is_invincible', 'false')

    # Set the rolename
    if blueprint.has_attribute('role_name'):
        blueprint.set_attribute('role_name', rolename)

    return blueprint


def handle_actor_batch(batch, tick=True):
    """
    Forward a CARLA command batch to spawn actors to CARLA, and gather the responses.
    Returns the list of actors that were spawned
    """
    client = SimulationData.get_client()
    if client is None:
        raise ValueError("class member 'client' not initialized yet")

    responses = client.apply_batch_sync(batch, tick)
    _wait_for_actors(tick)

    actor_ids = [r.actor_id for r in responses if not r.error]
    for r in responses:
        if r.error:
            print("WARNING: Not all actors were spawned")
            break
    return list(SimulationData.get_world().get_actors(actor_ids))


def spawn_actor(bp, spawn_point, must_spawn=False, attach_to=None,
                attachment_type=carla.AttachmentType.Rigid):
    # type: (carla.ActorBlueprint, carla.Waypoint | carla.Transform, bool, carla.Actor | None, carla.AttachmentType) -> carla.Actor | None # pylint: disable=line-too-long
    """
    Spawn an actor from an already built blueprint, optionally attached to a parent.

    Args:
        bp (carla.ActorBlueprint): The blueprint of the actor to spawn.
        spawn_point (carla.Transform | carla.Waypoint): Where to spawn it.
        must_spawn (bool, optional):
            If True, the actor will be spawned or an exception will be raised.
            If False, the function returns None if the actor could not be spawned.
            Defaults to False.
        attach_to (carla.Actor | None, optional):
            The parent object that the spawned actor will follow around.
            Defaults to None.
        attachment_type (carla.AttachmentType, optional):
            Determines how fixed and rigorous should be the changes in position
            according to its parent object.
            Defaults to carla.AttachmentType.Rigid.

    Returns:
        carla.Actor | None: The spawned actor if successful, None otherwise.

    Raises:
        RuntimeError: if `must_spawn` is True and the actor could not be spawned.
    """
    if isinstance(spawn_point, carla.Waypoint):
        spawn_point = spawn_point.transform
    world = SimulationData.get_world()
    if must_spawn:
        return world.spawn_actor(bp, spawn_point, attach_to, attachment_type)
    return world.try_spawn_actor(bp, spawn_point, attach_to, attachment_type)


def request_new_actor(model, spawn_point=None, rolename='scenario', autopilot=False,
                      random_location=False, color=None, actor_category="car",
                      attribute_filter=None, tick=True):
    """
    This method tries to create a new actor, returning it if successful (None otherwise).
    """
    if not spawn_point and not random_location:
        raise ValueError("Either spawn_point or random_location must be set")
    elif spawn_point and random_location:
        print("INFO: request_new_actor : Both spawn_point and random_location are set. Omitting spawn_point.")
    blueprint = create_blueprint(model, rolename, color, actor_category, attribute_filter)

    world = SimulationData.get_world()

    if random_location:
        actor = None
        while not actor:
            spawn_point = SimulationData.get_rng().choice(SimulationData.get_spawn_points())
            actor = world.try_spawn_actor(blueprint, spawn_point)

    else:
        # For non prop models, slightly lift the actor to avoid collisions with the ground
        z_offset = 0.2 if 'prop' not in model else 0

        # DO NOT USE spawn_point directly, as this will modify spawn_point permanently
        _spawn_point = carla.Transform(carla.Location(), spawn_point.rotation)
        _spawn_point.location.x = spawn_point.location.x
        _spawn_point.location.y = spawn_point.location.y
        _spawn_point.location.z = spawn_point.location.z + z_offset
        actor = world.try_spawn_actor(blueprint, _spawn_point)

    if actor is None:
        print("WARNING: Cannot spawn actor {} at position {}".format(model, spawn_point.location))
        return None

    # De/activate the autopilot of the actor if it belongs to vehicle
    if autopilot:
        if isinstance(actor, carla.Vehicle):
            actor.set_autopilot(autopilot, SimulationData.get_traffic_manager_port())
        else:
            print("WARNING: Tried to set the autopilot of a non vehicle actor")

    # Wait for the actor to be spawned properly before we do anything
    _wait_for_actors(tick)

    return actor


def request_new_actors(actor_list, attribute_filter=None, tick=True):
    """
    This method tries to series of actor in batch. If this was successful,
    the new actors are returned, None otherwise.

    param:
    - actor_list: list of ActorConfigurationData
    """
    SpawnActor = carla.command.SpawnActor  # pylint: disable=invalid-name
    PhysicsCommand = carla.command.SetSimulatePhysics  # pylint: disable=invalid-name
    FutureActor = carla.command.FutureActor  # pylint: disable=invalid-name
    ApplyTransform = carla.command.ApplyTransform  # pylint: disable=invalid-name
    SetAutopilot = carla.command.SetAutopilot  # pylint: disable=invalid-name
    SetVehicleLightState = carla.command.SetVehicleLightState  # pylint: disable=invalid-name

    tm_port = SimulationData.get_traffic_manager_port()
    map_name = SimulationData.get_map().name.split("/")[-1]
    batch = []

    SimulationData.generate_spawn_points()

    for actor in actor_list:
        # Get the blueprint
        blueprint = create_blueprint(
            actor.model, actor.rolename, actor.color, actor.category, attribute_filter
        )

        # Get the spawn point
        transform = actor.transform
        if not transform:
            continue
        if actor.random_location:
            _spawn_point = SimulationData.next_spawn_point()
            if _spawn_point is None:
                print("No more spawn points to use")
                break
        else:
            _spawn_point = carla.Transform()
            _spawn_point.rotation = transform.rotation
            _spawn_point.location.x = transform.location.x
            _spawn_point.location.y = transform.location.y
            if blueprint.has_tag('walker') and map_name.startswith('OpenDrive'):
                # On imported OpenDRIVE maps, spawning of pedestrians can fail.
                # By increasing the z-value the chances of success are increased.
                _spawn_point.location.z = transform.location.z + 0.8
            else:
                _spawn_point.location.z = transform.location.z + 0.2

        # Get the command
        command = SpawnActor(blueprint, _spawn_point)
        command.then(SetAutopilot(FutureActor, actor.autopilot, tm_port))

        if actor.args is not None and 'physics' in actor.args and actor.args['physics'] == "off":
            command.then(ApplyTransform(FutureActor, _spawn_point)).then(PhysicsCommand(FutureActor, False))
        elif actor.category == 'misc':
            command.then(PhysicsCommand(FutureActor, True))
        if actor.args is not None and 'lights' in actor.args and actor.args['lights'] == "on":
            command.then(SetVehicleLightState(FutureActor, carla.VehicleLightState.All))

        batch.append(command)

    actors = handle_actor_batch(batch, tick)

    if not actors:
        return None

    return actors


def request_new_batch_actors(model, amount, spawn_points, autopilot=False, random_location=False,
                             rolename='scenario', attribute_filter=None, tick=True):
    """
    Simplified version of "request_new_actors". This method also create several actors in batch.

    Instead of needing a list of ActorConfigurationData, an "amount" parameter is used.
    This makes actor spawning easier but reduces the amount of configurability.

    Some parameters are the same for all actors (rolename, autopilot and random location)
    while others are randomized (color)
    """
    SpawnActor = carla.command.SpawnActor  # pylint: disable=invalid-name
    SetAutopilot = carla.command.SetAutopilot  # pylint: disable=invalid-name
    FutureActor = carla.command.FutureActor  # pylint: disable=invalid-name

    tm_port = SimulationData.get_traffic_manager_port()
    batch = []

    SimulationData.generate_spawn_points()

    for i in range(amount):
        # Get vehicle by model
        blueprint = create_blueprint(model, rolename, attribute_filter=attribute_filter)

        if random_location:
            spawn_point = SimulationData.next_spawn_point()
            if spawn_point is None:
                print("No more spawn points to use. Spawned {} actors out of {}".format(i + 1, amount))
                break
        else:
            try:
                spawn_point = spawn_points[i]
            except IndexError:
                print("The amount of spawn points is lower than the amount of vehicles spawned")
                break

        if spawn_point:
            batch.append(
                SpawnActor(blueprint, spawn_point).then(SetAutopilot(FutureActor, autopilot, tm_port))
            )

    return handle_actor_batch(batch, tick)
