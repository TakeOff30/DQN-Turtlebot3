#!/usr/bin/env python3
"""
<<<<<<< Updated upstream
Moving Obstacles Node for Stage 3 Training.

Moves cylinder models along waypoint paths by publishing to /gazebo/set_model_state.
Uses simulation time (/clock) so movement pauses/resumes correctly with
gazebo.pauseSim()/unpauseSim() during DQN training.

Handles simulation resets (time jumping to 0) gracefully — this is critical
because the training loop calls resetSim() between every episode.

Models must exist in the Gazebo world as <model> (not <actor>) with
kinematic=true links for proper lidar detection and collision.
=======
Moving Obstacles Node - Random Walk
Moves cylinder models randomly within the arena.
>>>>>>> Stashed changes
"""

import rospy
import math
import random
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Quaternion


class RandomWalker:
    """Manages random movement state for a single obstacle."""

    def __init__(self, name, start_pos, speed=0.1):
        self.name = name
        self.x, self.y, self.z = start_pos
        self.speed = speed
        self.yaw = random.uniform(-math.pi, math.pi)
        
        self.time_since_change = 0.0
        self.change_interval = 5.0
        self.limit = 2.8 # Stage 4 is 6x6, keeping buffer from 3.0 walls

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

<<<<<<< Updated upstream
        # Loop time
        t_mod = t % self.total_time

        # Find which segment we're on
        elapsed = 0.0
        for i, seg_time in enumerate(self.segment_times):
            if elapsed + seg_time >= t_mod:
                # Interpolate within this segment
                frac = (t_mod - elapsed) / seg_time if seg_time > 0 else 0
                x0, y0, z0 = self.waypoints[i]
                x1, y1, z1 = self.waypoints[i + 1]

                x = x0 + frac * (x1 - x0)
                y = y0 + frac * (y1 - y0)
                z = z0 + frac * (z1 - z0)

                # Yaw: face direction of travel
                yaw = math.atan2(y1 - y0, x1 - x0)
                return x, y, z, yaw

            elapsed += seg_time

        # Fallback
        wp = self.waypoints[0]
        return wp[0], wp[1], wp[2], 0.0
=======
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
>>>>>>> Stashed changes


def yaw_to_quaternion(yaw):
    """Convert yaw angle to geometry_msgs Quaternion (z-axis rotation)."""
    return Quaternion(
        x=0.0,
        y=0.0,
        z=math.sin(yaw / 2.0),
        w=math.cos(yaw / 2.0)
    )


<<<<<<< Updated upstream
def main():
    rospy.init_node('moving_obstacles_node', anonymous=False)

    # Movement speed (m/s)
    speed = rospy.get_param('~obstacle_speed', 0.15)
    # Update rate (Hz) - how often we publish new poses
=======
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


def main():
    rospy.init_node('moving_obstacles_node', anonymous=False)

    # Global speed setting (can be overridden by params)
    default_speed = rospy.get_param('~obstacle_speed', 0.1) 
>>>>>>> Stashed changes
    rate_hz = rospy.get_param('~update_rate', 30)

    # Use a TOPIC publisher instead of a service call.
    # Publishing is non-blocking and more robust during rapid pause/unpause
    # cycles that happen every training step.
    pub = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=10)

<<<<<<< Updated upstream
    # Define waypoint paths (closed loops, last point = first point)
    # Obstacle 1: Clockwise rectangular path
    path1 = WaypointPath([
        (-1.2, -1.2, 0.3),
        (1.2, -1.2, 0.3),
        (1.2, 1.2, 0.3),
        (-1.2, 1.2, 0.3),
        (-1.2, -1.2, 0.3),
    ], speed=speed)

    # Obstacle 2: Counter-clockwise rectangular path
    path2 = WaypointPath([
        (1.2, 1.2, 0.3),
        (-1.2, 1.2, 0.3),
        (-1.2, -1.2, 0.3),
        (1.2, -1.2, 0.3),
        (1.2, 1.2, 0.3),
    ], speed=speed)

    obstacles = [
        ('moving_obstacle_1', path1),
        ('moving_obstacle_2', path2),
    ]

    rate = rospy.Rate(rate_hz)
    last_sim_time = 0.0

    rospy.loginfo(
        f"[MovingObstacles] Moving {len(obstacles)} obstacles at {speed} m/s, "
        f"{rate_hz} Hz update rate"
    )
=======
    obstacles = load_obstacles(default_speed)
    rospy.loginfo(f"[RandomWalk] Controlling {len(obstacles)} obstacles. Speed={default_speed} m/s")

    rate = rospy.Rate(rate_hz)
    last_time = rospy.get_time()
>>>>>>> Stashed changes

    while not rospy.is_shutdown():
        try:
            now = rospy.get_time()
            dt = now - last_time
            last_time = now

<<<<<<< Updated upstream
            # Skip if sim time is 0 (Gazebo not yet publishing /clock or just reset)
            if now < 0.001:
                rospy.sleep(0.05)
                continue

            # Detect time reset (resetSim was called between episodes)
            if now < last_sim_time:
                rospy.loginfo(
                    f"[MovingObstacles] Sim time reset detected "
                    f"({last_sim_time:.1f} -> {now:.1f}), continuing..."
                )

            last_sim_time = now

            for model_name, path in obstacles:
                x, y, z, yaw = path.get_pose(now)
=======
            # Handle simulation reset or pause (negative or huge dt)
            if dt < 0 or dt > 1.0:
                continue

            for walker in obstacles:
                x, y, z, yaw = walker.update(dt)
>>>>>>> Stashed changes

                state = ModelState()
                state.model_name = walker.name
                state.reference_frame = 'world'
                state.pose.position.x = x
                state.pose.position.y = y
                state.pose.position.z = z
                state.pose.orientation = yaw_to_quaternion(yaw)

                pub.publish(state)

            rate.sleep()

<<<<<<< Updated upstream
        except rospy.ROSTimeMovedBackwardsException:
            # Thrown by rate.sleep() when resetSim() jumps sim time to 0.
            # Just continue — the next iteration will pick up the new time.
            rospy.loginfo("[MovingObstacles] Time moved backwards (sim reset), recovering...")
            continue

=======
        except rospy.ROSInterruptException:
            pass
>>>>>>> Stashed changes

if __name__ == '__main__':
    main()
