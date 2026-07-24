#!/usr/bin/env python
#
# Copyright (c) Institute for Automotive Engineering (ika), RWTH Aachen University
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
"""
ROS Vehicle Control that sends route action from scenario
"""
import math
import threading

import rclpy
from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Time
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node

from geometry_msgs.msg import PointStamped, Point
from route_planning_msgs.action import PlanRoute
from trajectory_planning_msgs.msg import Trajectory

import carla

from srunner.scenariomanager.actorcontrols.basic_control import BasicControl  # pylint: disable=import-error
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider


class RosVehicleControlRouteAction(BasicControl):
    def __init__(self, actor, args=None):
        super().__init__(actor)

        args = args or {}
        print(f"RosVehicleControlRouteAction args: {args}", flush=True)

        params = {}
        params["trajectory_topic"] = "/planning/drivable_trajectory"
        params["route_action"] = "/planning/lanelet2_route_planning/plan_route"

        self._initial_speed_duration = float(args.get("initial_speed_duration", 0.5))
        self._initial_speed_end_time = None
        self._route_action_offset = float(args.get("route_action_offset", 0.2))
        self._route_action_time = None
        self._debug_time_offset = float(args.get("debug_time_offset", 0.0))

        if "trajectory_topic" in args:
            params["trajectory_topic"] = args["trajectory_topic"]

        if "route_action" in args:
            params["route_action"] = args["route_action"]

        if not rclpy.ok():
            rclpy.init()

        self.node = NavigationClient(params)
        self.node.get_logger().info(
            f"Route ROS client initialized "
            f"(trajectory_topic='{params['trajectory_topic']}', "
            f"route_action='{params['route_action']}')"
        )

        # Run ROS 2 spinning in a separate thread to avoid blocking
        self.ros_thread = threading.Thread(target=rclpy.spin, args=(self.node,), daemon=True)
        self.ros_thread.start()
        self.node.get_logger().info("Started ROS 2 spinning thread for NavigationClient")

    def reset(self):
        pass

    def run_step(self):
        time = self._get_sim_time()

        self.check_requirements()

        # This is a debugging workaround for visualization only.
        if time < self._debug_time_offset:
            return

        # Set timestamps for initial speed and route action if not already set
        if self._initial_speed_end_time is None:
            self._initial_speed_end_time = time + self._initial_speed_duration
            self._route_action_time = time + self._route_action_offset

        # Trigger the route action at the specified time
        if self._route_action_time is not None and time >= self._route_action_time:
            self._route_action_time = None

            self.node.call_route_action()
            CarlaDataProvider.register_route_action_client()

        # Apply initial speed until the end time is reached
        if time < self._initial_speed_end_time:
            self._apply_initial_speed()

    def check_requirements(self):

        if not CarlaDataProvider.is_scenario_running():
            return

        if not self.node.goal_pose:
            return

        if not self.node.stack_ready:
            return

        if self.node.reached_goal:
            return

    def update_waypoints(self, waypoints, start_time=None):
        self.node.set_goal_pose(waypoints)
        self._initial_speed_end_time = None
        self._route_action_time = None
        return super().update_waypoints(waypoints, start_time)

    def update_final_route_goal(self, waypoint):
        """Set the exact OpenSCENARIO endpoint independently of the planned route."""
        self.node.set_goal_pose([waypoint])

    def check_reached_waypoint_goal(self):
        return self.node.reached_goal

    def _apply_initial_speed(self):
        if self._target_speed <= 0:
            return

        yaw = self._actor.get_transform().rotation.yaw * (math.pi / 180)
        vx = math.cos(yaw) * self._target_speed
        vy = math.sin(yaw) * self._target_speed
        self._actor.set_target_velocity(carla.Vector3D(vx, vy, 0))

    def _get_sim_time(self):
        world = CarlaDataProvider.get_world()
        if world is None:
            return None

        snapshot = world.get_snapshot()
        if snapshot is None:
            return None

        return snapshot.timestamp.elapsed_seconds


class NavigationClient(Node):
    def __init__(self, params):
        super().__init__('ros_route_agent')

        self.goal_pose = None
        self.reached_goal = False
        self.stack_ready = False
        self.route_triggered_flag = False

        self.trajectory_sub = self.create_subscription(
            Trajectory,
            params["trajectory_topic"],
            self.trajectory_callback,
            10
        )

        self.route_action_client = ActionClient(self, PlanRoute, params["route_action"])

        # wait for the action server to be available
        self.get_logger().info(
            f"Subscribing to trajectory topic '{params['trajectory_topic']}' "
            f"and waiting for action server '{params['route_action']}'"
        )
        self.route_action_client.wait_for_server()
        self.get_logger().info(f"Route action server '{params['route_action']}' available")

    def trajectory_callback(self, msg):

        if not self.stack_ready:
            self.stack_ready = True
            self.get_logger().info("Received first trajectory, stack ready")

    def set_goal_pose(self, waypoints):
        """Set the goal pose from waypoints"""
        self.goal_pose = PointStamped()
        self.goal_pose.header.frame_id = "carla_map"
        self.goal_pose.header.stamp = Time(sec=0, nanosec=0)

        self.goal_pose.point = Point(
            x=waypoints[-1].location.x,
            y=-waypoints[-1].location.y,
            z=0.0
        )
        # Ensure a new goal can trigger the action again
        self.route_triggered_flag = False
        self.reached_goal = False

    def call_route_action(self):
        """Send the goal_pose to the action server"""
        if self.route_triggered_flag:
            self.get_logger().info("Route action already triggered, skipping")
            return

        self.get_logger().info(f"Sending goal destination ({self.goal_pose.point.x}, {self.goal_pose.point.y})")

        goal_msg = PlanRoute.Goal()
        goal_msg.destination = self.goal_pose

        send_goal_future = self.route_action_client.send_goal_async(goal_msg, feedback_callback=self.feedback_callback)
        send_goal_future.add_done_callback(self.action_response_callback)
        self.route_triggered_flag = True

    def action_response_callback(self, future):
        """Called when the goal is accepted or rejected"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warning("Route action goal rejected")
            self.route_triggered_flag = False
            return

        self.get_logger().info("Route action goal accepted")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.result_callback)

    def feedback_callback(self, feedback_msg):
        """Process feedback from the action server"""
        feedback = feedback_msg.feedback
        self.get_logger().debug(
            f"Route action feedback: distance remaining = {feedback.distance_remaining}"
        )

    def result_callback(self, future):
        """Called when the goal is completed"""
        result = future.result()

        self.route_triggered_flag = False

        if result is not None and result.status == GoalStatus.STATUS_SUCCEEDED:
            self.reached_goal = True
            CarlaDataProvider.mark_route_action_completed()
            
        self.get_logger().info(f"Route action goal completed with status: {result}")
        self.get_logger().info("Shutting down rclpy after route result callback")
        rclpy.shutdown()
