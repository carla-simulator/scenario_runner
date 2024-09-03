"""Behaviors for dynamic agents in CARLA scenarios."""

from scenic.domains.driving.behaviors import *	# use common driving behaviors

try:
    from scenic.simulators.carla.actions import *
except ModuleNotFoundError:
    pass    # ignore; error will be caught later if user attempts to run a simulation

behavior AutopilotBehavior():
    """Behavior causing a vehicle to use CARLA's built-in autopilot."""
    take SetAutopilotAction(True)

behavior WalkForwardBehavior(speed=0.5):
    take SetWalkingDirectionAction(self.heading), SetWalkingSpeedAction(speed)

behavior WalkBehavior(maxSpeed=1.4):
    take SetWalkAction(True, maxSpeed)

behavior CrossingBehavior(reference_actor, min_speed=1, threshold=10, final_speed=None):
    """
    This behavior dynamically controls the speed of an actor that will perpendicularly (or close to)
    cross the road, so that it arrives at a spot in the road at the same time as a reference actor.

    Args:
        min_speed (float): minimum speed of the crossing actor. As this is a type of "synchronization action",
            a minimum speed is needed, to allow the actor to keep moving even if the reference actor has stopped
        threshold (float): starting distance at which the crossing actor starts moving
        final_speed (float): speed of the crossing actor after the reference one surpasses it
    """

    if not final_speed:
        final_speed = min_speed

    while (distance from self to reference_actor) > threshold:
        wait

    while True:
        distance_vec = self.position - reference_actor.position
        rotated_vec = distance_vec.rotatedBy(-reference_actor.heading)

        ref_dist = rotated_vec.y
        if ref_dist < 0:
            # The reference_actor has passed the crossing object, no need to keep monitoring the speed
            break

        actor_dist = rotated_vec.x

        ref_speed = reference_actor.speed
        ref_time = ref_speed / ref_dist

        actor_speed = actor_dist * ref_time
        if actor_speed < min_speed:
            actor_speed = min_speed

        if isinstance(self, Walks):
            do WalkForwardBehavior(actor_speed)
        elif isinstance(self, Steers):
            take SetSpeedAction(actor_speed)

    if isinstance(self, Walks):
        do WalkForwardBehavior(final_speed)
    elif isinstance(self, Steers):
        take SetSpeedAction(final_speed)
