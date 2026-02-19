#!/usr/bin/env python3
"""
Moving Obstacles Node - Random Walk
Moves cylinder models randomly within the arena.
"""

import rospy
import math
import random
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Quaternion
from std_msgs.msg import Empty


class RandomWalker:
    """Manages random movement state for a single obstacle."""

    def __init__(self, name, start_pos, speed=0.1):
        self.name = name
        self.initial_pos = start_pos  # Store initial position for resets
        self.x, self.y, self.z = start_pos
        self.speed = speed
        self.yaw = random.uniform(-math.pi, math.pi)
        
        self.time_since_change = 0.0
        self.change_interval = 5.0
        self.limit = 2.8 # Stage 4 is 6x6, keeping buffer from 3.0 walls

    def reset(self):
        """Reset to initial position and random orientation."""
        self.x, self.y, self.z = self.initial_pos
        self.yaw = random.uniform(-math.pi, math.pi)
        self.time_since_change = 0.0
        rospy.loginfo(f"[{self.name}] Reset to initial position ({self.x:.2f}, {self.y:.2f})")

    def update(self, dt):
        """Update position based on speed, yaw, and time delta."""
        self.time_since_change += dt

        # Change direction every 5 seconds
        if self.time_since_change >= self.change_interval:
            self.yaw = random.uniform(-math.pi, math.pi)
            self.time_since_change = 0.0
            rospy.logdebug(f"[{self.name}] Changing direction")

        # Move
        dx = self.speed * math.cos(self.yaw) * dt
        dy = self.speed * math.sin(self.yaw) * dt

        next_x = self.x + dx
        next_y = self.y + dy

        # Boundary checks (Bouncing)
        if next_x < -self.limit or next_x > self.limit:
            next_x = max(-self.limit, min(self.limit, next_x))
            self.yaw = math.pi - self.yaw # Reflect across X-axis
        
        if next_y < -self.limit or next_y > self.limit:
            next_y = max(-self.limit, min(self.limit, next_y))
            self.yaw = -self.yaw # Reflect across Y-axis

        self.x = next_x
        self.y = next_y

        return self.x, self.y, self.z, self.yaw


def yaw_to_quaternion(yaw):
    """Convert yaw angle to geometry_msgs Quaternion (z-axis rotation)."""
    return Quaternion(
        x=0.0,
        y=0.0,
        z=math.sin(yaw / 2.0),
        w=math.cos(yaw / 2.0)
    )


def load_obstacles(default_speed):
    """Load obstacles from params, using first waypoint as start pos."""
    paths_param = rospy.get_param('~paths', None)
    obstacles = []

    if paths_param:
        for obs_def in paths_param:
            name = obs_def['name']
            # Use first waypoint as start position
            if 'waypoints' in obs_def and len(obs_def['waypoints']) > 0:
                start_pos = tuple(obs_def['waypoints'][0])
            else:
                start_pos = (0, 0, 0.3)
            
            # Use param speed if specific, else default
            spd = obs_def.get('speed', default_speed)
            obstacles.append(RandomWalker(name, start_pos, spd))
    else:
        # Fallback if no params
        obstacles.append(RandomWalker('moving_obstacle_1', (1.0, 0.0, 0.3), default_speed))
        obstacles.append(RandomWalker('moving_obstacle_2', (-1.0, 0.0, 0.3), default_speed))
    
    return obstacles


def perform_reset(obstacles, pub):
    """Reset all walkers and publish their initial states."""
    rospy.loginfo("[RandomWalk] Resetting obstacles to initial positions")
    for walker in obstacles:
        walker.reset()
        # Immediately publish reset positions
        state = ModelState()
        state.model_name = walker.name
        state.reference_frame = 'world'
        state.pose.position.x = walker.x
        state.pose.position.y = walker.y
        state.pose.position.z = walker.z
        state.pose.orientation = yaw_to_quaternion(walker.yaw)
        pub.publish(state)

def main():
    rospy.init_node('moving_obstacles_node', anonymous=False)

    # Global speed setting (can be overridden by params)
    default_speed = rospy.get_param('~obstacle_speed', 0.1) 
    rate_hz = rospy.get_param('~update_rate', 30)

    # Use a TOPIC publisher instead of a service call.
    # Publishing is non-blocking and more robust during rapid pause/unpause
    # cycles that happen every training step.
    pub = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=10)

    obstacles = load_obstacles(default_speed)
    rospy.loginfo(f"[RandomWalk] Controlling {len(obstacles)} obstacles. Speed={default_speed} m/s")

    reset_requested = False

    def handle_episode_reset(_msg):
        nonlocal reset_requested
        reset_requested = True

    rospy.Subscriber('/moving_obstacles/reset', Empty, handle_episode_reset, queue_size=1)

    rate = rospy.Rate(rate_hz)
    last_time = rospy.get_time()
    last_sim_time = 0.0  # Track sim time for reset detection

    while not rospy.is_shutdown():
        try:
            now = rospy.get_time()
            dt = now - last_time
            last_time = now

            # Detect simulation reset (time jump backwards or to near-zero)
            if now < last_sim_time or (now < 1.0 and last_sim_time > 1.0):
                perform_reset(obstacles, pub)
                last_sim_time = now
                continue

            if reset_requested:
                perform_reset(obstacles, pub)
                reset_requested = False

            last_sim_time = now

            # Handle simulation pause (negative or huge dt)
            if dt < 0 or dt > 1.0:
                continue

            for walker in obstacles:
                x, y, z, yaw = walker.update(dt)

                state = ModelState()
                state.model_name = walker.name
                state.reference_frame = 'world'
                state.pose.position.x = x
                state.pose.position.y = y
                state.pose.position.z = z
                state.pose.orientation = yaw_to_quaternion(yaw)

                pub.publish(state)

            rate.sleep()

        except rospy.ROSTimeMovedBackwardsException:
            rospy.logwarn("[RandomWalk] Time moved backwards (sim reset), resetting obstacles...")
            perform_reset(obstacles, pub)
            # Reset time trackers to avoid huge dt on next loop
            last_time = rospy.get_time()
            last_sim_time = last_time
            continue
        except rospy.ROSInterruptException:
            pass

if __name__ == '__main__':
    main()

