#!/usr/bin/env python3
"""
Moving Obstacles Node — Modular, param-driven.

Moves cylinder models along waypoint paths by publishing to /gazebo/set_model_state.
Uses simulation time (/clock) so movement pauses/resumes correctly with
gazebo.pauseSim()/unpauseSim() during DQN training.

Obstacle paths are read from ROS params (~paths). If no params are found,
falls back to default Stage 3 rectangular paths for backward compatibility.

Param format (YAML inside launch file's <rosparam> block):
    paths:
      - name: moving_obstacle_1
        speed: 0.15          # optional per-obstacle speed override
        waypoints:
          - [0.0, -2.5, 0.3]
          - [0.0,  2.0, 0.3]
      - name: moving_obstacle_2
        waypoints: [...]

Models must exist in the Gazebo world as <model> (not <actor>) with
kinematic=true links for proper lidar detection and collision.
"""

import rospy
import math
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Quaternion
from std_msgs.msg import Empty


class WaypointPath:
    """Defines a looping waypoint path with linear interpolation."""

    def __init__(self, waypoints, speed=0.15):
        """
        Args:
            waypoints: List of (x, y, z) tuples forming a closed loop.
                       Last point should equal first for seamless looping.
            speed: Movement speed in m/s.
        """
        self.waypoints = waypoints
        self.speed = speed

        # Precompute segment lengths and cumulative times
        self.segment_lengths = []
        self.segment_times = []
        total_time = 0.0

        for i in range(len(waypoints) - 1):
            dx = waypoints[i + 1][0] - waypoints[i][0]
            dy = waypoints[i + 1][1] - waypoints[i][1]
            length = math.sqrt(dx * dx + dy * dy)
            self.segment_lengths.append(length)
            seg_time = length / speed if speed > 0 else 0
            self.segment_times.append(seg_time)
            total_time += seg_time

        self.total_time = total_time

    def get_pose(self, t):
        """Get interpolated (x, y, z, yaw) at time t (loops automatically)."""
        if self.total_time <= 0:
            wp = self.waypoints[0]
            return wp[0], wp[1], wp[2], 0.0

        t_mod = t % self.total_time

        elapsed = 0.0
        for i, seg_time in enumerate(self.segment_times):
            if elapsed + seg_time >= t_mod:
                frac = (t_mod - elapsed) / seg_time if seg_time > 0 else 0
                x0, y0, z0 = self.waypoints[i]
                x1, y1, z1 = self.waypoints[i + 1]

                x = x0 + frac * (x1 - x0)
                y = y0 + frac * (y1 - y0)
                z = z0 + frac * (z1 - z0)

                yaw = math.atan2(y1 - y0, x1 - x0)
                return x, y, z, yaw

            elapsed += seg_time

        wp = self.waypoints[0]
        return wp[0], wp[1], wp[2], 0.0


def yaw_to_quaternion(yaw):
    """Convert yaw angle to geometry_msgs Quaternion (z-axis rotation)."""
    return Quaternion(
        x=0.0,
        y=0.0,
        z=math.sin(yaw / 2.0),
        w=math.cos(yaw / 2.0)
    )


def load_obstacles_from_params(default_speed):
    """
    Load obstacle definitions from ROS params (~paths).

    Returns list of (model_name, WaypointPath) tuples.
    Falls back to default Stage 3 rectangular paths if no params found.
    """
    paths_param = rospy.get_param('~paths', None)

    if paths_param is not None:
        obstacles = []
        for obs_def in paths_param:
            name = obs_def['name']
            waypoints = [tuple(wp) for wp in obs_def['waypoints']]
            obs_speed = obs_def.get('speed', default_speed)
            path = WaypointPath(waypoints, speed=obs_speed)
            obstacles.append((name, path))
            rospy.loginfo(
                "[MovingObstacles] Loaded '{}': {} waypoints, {} m/s, "
                "loop time {:.1f}s".format(name, len(waypoints), obs_speed, path.total_time)
            )
        return obstacles

    rospy.loginfo("[MovingObstacles] No ~paths param found, using default Stage 3 paths")
    path1 = WaypointPath([
        (-1.2, -1.2, 0.3),
        ( 1.2, -1.2, 0.3),
        ( 1.2,  1.2, 0.3),
        (-1.2,  1.2, 0.3),
        (-1.2, -1.2, 0.3),
    ], speed=default_speed)

    path2 = WaypointPath([
        ( 1.2,  1.2, 0.3),
        (-1.2,  1.2, 0.3),
        (-1.2, -1.2, 0.3),
        ( 1.2, -1.2, 0.3),
        ( 1.2,  1.2, 0.3),
    ], speed=default_speed)

    return [
        ('moving_obstacle_1', path1),
        ('moving_obstacle_2', path2),
    ]


def publish_obstacles_at_path_time(pub, obstacles, path_time):
    for model_name, path in obstacles:
        x, y, z, yaw = path.get_pose(path_time)

        state = ModelState()
        state.model_name = model_name
        state.reference_frame = 'world'
        state.pose.position.x = x
        state.pose.position.y = y
        state.pose.position.z = z
        state.pose.orientation = yaw_to_quaternion(yaw)

        pub.publish(state)


def main():
    rospy.init_node('moving_obstacles_node', anonymous=False)

    default_speed = rospy.get_param('~obstacle_speed', 0.15)
    rate_hz = rospy.get_param('~update_rate', 30)

    pub = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=10)

    obstacles = load_obstacles_from_params(default_speed)

    path_time_offset = 0.0

    def handle_reset(_msg):
        nonlocal path_time_offset
        now = rospy.get_time()
        path_time_offset = now
        publish_obstacles_at_path_time(pub, obstacles, 0.0)
        rospy.loginfo("[MovingObstacles] Episode reset received, obstacles reset to initial waypoints")

    rospy.Subscriber('/moving_obstacles/reset', Empty, handle_reset, queue_size=1)

    rate = rospy.Rate(rate_hz)
    last_sim_time = 0.0

    rospy.loginfo(
        "[MovingObstacles] Moving {} obstacles, {} Hz update rate".format(
            len(obstacles), rate_hz)
    )

    while not rospy.is_shutdown():
        try:
            now = rospy.get_time()

            if now < 0.001:
                rospy.sleep(0.05)
                continue

            if now < last_sim_time:
                rospy.loginfo(
                    "[MovingObstacles] Sim time reset detected "
                    "({:.1f} -> {:.1f}), continuing...".format(last_sim_time, now)
                )

            last_sim_time = now

            path_time = now - path_time_offset
            publish_obstacles_at_path_time(pub, obstacles, path_time)

            rate.sleep()

        except rospy.ROSTimeMovedBackwardsException:
            rospy.loginfo("[MovingObstacles] Time moved backwards (sim reset), recovering...")
            continue


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
