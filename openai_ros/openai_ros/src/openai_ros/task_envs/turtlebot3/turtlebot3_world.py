import rospy
import numpy
from gym import spaces
from openai_ros.robot_envs import turtlebot3_env
from geometry_msgs.msg import Quaternion
from openai_ros.task_envs.task_commons import LoadYamlFileParamsTest
from openai_ros.openai_ros_common import ROSLauncher
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
from std_msgs.msg import Empty
import os
import random
import math

class TurtleBot3WorldEnv(turtlebot3_env.TurtleBot3Env):
    def __init__(self):
        """
        This Task Env is designed for having the TurtleBot3 in the turtlebot3 world
        closed room with columns.
        It will learn how to move around without crashing.
        """
        # This is the path where the simulation files, the Task and the Robot gits will be downloaded if not there
        ros_ws_abspath = rospy.get_param("/turtlebot3/ros_ws_abspath", None)
        assert ros_ws_abspath is not None, "You forgot to set ros_ws_abspath in your yaml file of your main RL script. Set ros_ws_abspath: \'YOUR/SIM_WS/PATH\'"
        assert os.path.exists(ros_ws_abspath), "The Simulation ROS Workspace path " + ros_ws_abspath + \
                                               " DOESNT exist, execute: mkdir -p " + ros_ws_abspath + \
                                               "/src;cd " + ros_ws_abspath + ";catkin_make"

        ROSLauncher(rospackage_name="turtlebot3_gazebo",
                    launch_file_name=rospy.get_param('/turtlebot3/launch_file_name', 'start_world.launch'),
                    ros_ws_abspath=ros_ws_abspath)

        # Load Params from the desired Yaml file
        LoadYamlFileParamsTest(rospackage_name="openai_ros",
                               rel_path_from_package_to_file="src/openai_ros/task_envs/turtlebot3/config",
                               yaml_file_name="turtlebot3_world.yaml")


        # Here we will add any init functions prior to starting the MyRobotEnv
        super(TurtleBot3WorldEnv, self).__init__(ros_ws_abspath)

        number_actions = rospy.get_param('/turtlebot3/n_actions')
        self.action_space = spaces.Discrete(number_actions)

        self.reward_range = (-numpy.inf, numpy.inf)

        # Actions and Observations
        self.linear_forward_speed = rospy.get_param('/turtlebot3/linear_forward_speed')
        self.linear_turn_speed = rospy.get_param('/turtlebot3/linear_turn_speed')
        self.angular_velocities = rospy.get_param('/turtlebot3/angular_velocities')
        self.init_linear_forward_speed = rospy.get_param('/turtlebot3/init_linear_forward_speed')
        self.init_linear_turn_speed = rospy.get_param('/turtlebot3/init_linear_turn_speed')
        self.angular_speed = 0

        self.new_ranges = rospy.get_param('/turtlebot3/new_ranges')
        self.min_range = rospy.get_param('/turtlebot3/min_range')
        self.max_laser_value = rospy.get_param('/turtlebot3/max_laser_value')
        self.min_laser_value = rospy.get_param('/turtlebot3/min_laser_value')
        self.max_linear_aceleration = rospy.get_param('/turtlebot3/max_linear_aceleration')
        
        self.max_episode_steps = rospy.get_param('/turtlebot3/max_episode_steps')
        
        # Arena boundaries for random spawning
        self.arena_min_x = rospy.get_param('/turtlebot3/arena_min_x', -1.5)
        self.arena_max_x = rospy.get_param('/turtlebot3/arena_max_x', 1.5)
        self.arena_min_y = rospy.get_param('/turtlebot3/arena_min_y', -1.5)
        self.arena_max_y = rospy.get_param('/turtlebot3/arena_max_y', 1.5)
        
        self.goal_positions = rospy.get_param('/turtlebot3/goal_positions', None)
        
        # Safe boundaries for goal spawning (inside walls)
        self.safe_arena_min_x = rospy.get_param('/turtlebot3/safe_arena_min_x', -1.3)
        self.safe_arena_max_x = rospy.get_param('/turtlebot3/safe_arena_max_x', 1.3)
        self.safe_arena_min_y = rospy.get_param('/turtlebot3/safe_arena_min_y', -1.3)
        self.safe_arena_max_y = rospy.get_param('/turtlebot3/safe_arena_max_y', 1.3)
        
        self.min_spawn_distance = rospy.get_param('/turtlebot3/min_spawn_distance', 1.5)
        self.success_threshold = rospy.get_param('/turtlebot3/success_threshold', 0.5)
        
        # Goal position - will be randomized in _init_env_variables
        self.goal_x = 0.0
        self.goal_y = 0.0
        
        # Reward parameters
        self.distance_reward_multiplier = rospy.get_param('/turtlebot3/distance_reward_multiplier', 50.0)
        self.turn_penalty_multiplier = rospy.get_param('/turtlebot3/turn_penalty_multiplier', 0.5)
        self.time_penalty = rospy.get_param('/turtlebot3/time_penalty', 0) # given at each step
        self.goal_reached_reward = rospy.get_param('/turtlebot3/goal_reached_reward', 300)
        self.obstacle_hit_penalty = rospy.get_param('/turtlebot3/obstacle_hit_penalty', -100)
        self.courage_zone_threshold = rospy.get_param('/turtlebot3/courage_zone_threshold', 0.5)
        self.yaw_reward_multiplier = rospy.get_param('/turtlebot3/yaw_reward_multiplier', 5)

        laser_scan = self.get_laser_scan()
        
        total_laser_readings = len(laser_scan.ranges)
        num_laser_readings = int(total_laser_readings / self.new_ranges)
    
        rospy.loginfo(f"Laser readings: {total_laser_readings} total, sampling every {self.new_ranges}th = {num_laser_readings} readings")
        laser_ranges, _ = self._compute_laser_scans(laser_scan)
        num_laser_readings = len(laser_ranges)
        
        # Calculate max possible distance within arena
        self.max_goal_distance = math.sqrt((self.arena_max_x - self.arena_min_x)**2 + 
                                           (self.arena_max_y - self.arena_min_y)**2)
        
        # declare bservation space: [laser_readings..., distance_to_goal, sin(angle), cos(angle)]
        laser_high = numpy.full((num_laser_readings,), self.max_laser_value, dtype=numpy.float32)
        laser_low = numpy.full((num_laser_readings,), self.min_laser_value, dtype=numpy.float32)
        obs_high = numpy.concatenate([laser_high, 
                                      numpy.array([self.max_goal_distance, 1.0, 1.0], dtype=numpy.float32)])
        obs_low = numpy.concatenate([laser_low, 
                                     numpy.array([0.0, -1.0, -1.0], dtype=numpy.float32)])
        
        obs_dim = num_laser_readings + 3  # laser + [distance_to_goal, sin(angle), cos(angle)]
        self.observation_space = spaces.Box(obs_low, obs_high, shape=(obs_dim,), dtype=numpy.float32)

        rospy.logdebug(f"ACTION SPACES TYPE {str(self.action_space)}")
        rospy.logdebug(f"OBSERVATION SPACES TYPE {str(self.observation_space)}")

        self.cumulated_steps = 0.0
        
        # in inference mode we end episode on third goal reached
        self.inference_mode = rospy.get_param('/turtlebot3/inference_mode', False)
        self.use_fixed_initial_pose = rospy.get_param('/turtlebot3/use_fixed_initial_pose', False)
        self.gazebo_model_name = rospy.get_param('/turtlebot3/gazebo_model_name', 'turtlebot3_burger')
        self.fixed_init_x = rospy.get_param('/turtlebot3/fixed_init_x', 0.0)
        self.fixed_init_y = rospy.get_param('/turtlebot3/fixed_init_y', 0.0)
        self.fixed_init_z = rospy.get_param('/turtlebot3/fixed_init_z', 0.0)
        self.fixed_init_yaw = rospy.get_param('/turtlebot3/fixed_init_yaw', 0.0)
        self.goals_reached_count = 0
        self.inference_goal_target_count = rospy.get_param('/turtlebot3/inference_goal_target_count', 3)
        self.inference_third_goal_x = rospy.get_param('/turtlebot3/inference_third_goal_x', 0.0)
        self.inference_third_goal_y = rospy.get_param('/turtlebot3/inference_third_goal_y', 2.2)
        
        # Initialize robot position tracking
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        self.previous_distance_to_goal = None
        self.succeed = False
        self.fail = False
        self._last_min_laser_value = None
        self._last_front_ranges = []
        self._last_front_angles = []
        
        # Wait for Gazebo service to move goal marker
        rospy.loginfo("Waiting for Gazebo set_model_state service...")
        rospy.wait_for_service('/gazebo/set_model_state')
        self.set_model_state_srv = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        rospy.loginfo("Gazebo service ready")

        # Trigger moving obstacles reset at every episode start (if obstacle node is running)
        self.reset_moving_obstacles_pub = rospy.Publisher('/moving_obstacles/reset', Empty, queue_size=1)
        self._move_goal_marker()

    def _update_goal_distance_reference(self):
        dx = self.goal_x - self.robot_x
        dy = self.goal_y - self.robot_y
        self.previous_distance_to_goal = math.sqrt(dx ** 2 + dy ** 2)

    def _set_robot_pose(self, x, y, z, yaw, model_name=None):
        state = ModelState()
        state.model_name = model_name or self.gazebo_model_name
        state.reference_frame = "world"
        state.pose.position.x = x
        state.pose.position.y = y
        state.pose.position.z = z
        state.pose.orientation = Quaternion(
            x=0.0,
            y=0.0,
            z=math.sin(yaw / 2.0),
            w=math.cos(yaw / 2.0),
        )
        self.set_model_state_srv(state)

    def _reset_robot_to_fixed_pose(self, stop_before_reset=False, log_message=None):
        if stop_before_reset:
            self.move_base(0.0, 0.0, epsilon=0.05, update_rate=10)
            rospy.sleep(0.1)

        self._set_robot_pose(
            self.fixed_init_x,
            self.fixed_init_y,
            self.fixed_init_z,
            self.fixed_init_yaw,
            self.gazebo_model_name,
        )

        if log_message:
            rospy.loginfo(log_message)

    def _move_goal_marker(self):
        """Move the existing goal marker to a new position using SetModelState"""
        angle = random.uniform(0, 2 * math.pi)  # Random direction angle
        distance = 1.0  # Fixed 1 meter distance in any direction
        
        if self.goal_positions:
            positions = random.choice(self.goal_positions)
            self.goal_x = positions[0]
            self.goal_y = positions[1]
        else:
            self.goal_x = self.robot_x + distance * math.cos(angle)
            self.goal_y = self.robot_y + distance * math.sin(angle)
            # Clamp to safe arena boundaries
            self.goal_x = numpy.clip(self.goal_x, self.safe_arena_min_x, self.safe_arena_max_x)
            self.goal_y = numpy.clip(self.goal_y, self.safe_arena_min_y, self.safe_arena_max_y)

        self._update_goal_distance_reference()
    
    def _position_goal_marker(self):
        """Move the goal marker to new position in Gazebo"""
        try:
            model_state = ModelState()
            model_state.model_name = 'goal_marker'
            model_state.pose.position.x = self.goal_x
            model_state.pose.position.y = self.goal_y
            model_state.pose.position.z = 0.1
            model_state.pose.orientation.w = 1.0
            
            self.set_model_state_srv(model_state)
            rospy.loginfo("Goal marker moved to (%.2f, %.2f)" % (self.goal_x, self.goal_y))
        except rospy.ServiceException as e:
            rospy.logerr("Failed to move goal marker: %s" % str(e))
        
    def _set_init_pose(self):
        """Sets the Robot in its init pose"""
        if self.inference_mode and self.use_fixed_initial_pose:
            self._reset_robot_to_fixed_pose(
                stop_before_reset=True,
                log_message=(
                    f"[INFERENCE] Spawning robot at fixed position: "
                    f"({self.fixed_init_x}, {self.fixed_init_y}, {self.fixed_init_z}), "
                    f"yaw={self.fixed_init_yaw}"
                ),
            )
        else:
            self.move_base(self.init_linear_forward_speed,
                        self.init_linear_turn_speed,
                        epsilon=0.05,
                        update_rate=10)

        return True

    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        of an episode. Generates random goal position within arena bounds.
        :return:
        """
        # Reset episode tracking
        self.succeed = False
        self.fail = False
        self.current_episode_step = 0
        self.goals_reached_count = 0

        # Reset moving cylinder obstacles to their initial waypoint at episode start
        self.reset_moving_obstacles_pub.publish(Empty())

        # In inference mode reposition robot in given initial pose
        if self.inference_mode and self.use_fixed_initial_pose:
            self._reset_robot_to_fixed_pose(
                stop_before_reset=False,
                log_message=f"[INFERENCE] Reset robot to fixed position: ({self.fixed_init_x}, {self.fixed_init_y})",
            )
        
    
        self._update_robot_position()
        self._move_goal_marker()  # Generate new random goal each episode
        self._position_goal_marker()  # Place marker in Gazebo
        self._update_goal_distance_reference()

    def _set_action(self, action):
        """
        This set action will Set the linear and angular speed of the turtlebot2
        based on the action number given.
        :param action: The action integer that set s what movement to do next.
        """
        
        self.angular_speed = self.angular_velocities[action]
        linear_speed = self.linear_forward_speed
        rospy.logwarn(f"ANGULAR VELOCITY: {self.angular_speed}")
        
        # We tell TurtleBot2 the linear and angular speed to set to execute
        self.move_base(linear_speed, self.angular_speed, epsilon=0.05, update_rate=10)
        
        # Increment episode step counter
        self.current_episode_step += 1

        rospy.logdebug("END Set Action ==>"+str(action))

    def _get_obs(self):
        """Computes laser scans and normalizes values for faster convergence"""
        rospy.logdebug("Start Get Observation ==>")
        
        self._update_robot_position()
        laser_scan = self.get_laser_scan()

        valid_ranges = [r for r in laser_scan.ranges if not (numpy.isinf(r) or numpy.isnan(r) or r == 0)]
        self._last_min_laser_value = min(valid_ranges) if len(valid_ranges) > 0 else None

        laser_ranges, _ = self._compute_laser_scans(laser_scan)
        self._last_front_ranges, self._last_front_angles = laser_ranges, _
        # Calculate relative goal information
        dx = self.goal_x - self.robot_x
        dy = self.goal_y - self.robot_y
        distance_to_goal = math.sqrt(dx**2 + dy**2)
        
        # Calculate angle to goal relative to robot's heading (Normalized to [-pi, pi])
        goal_angle = math.atan2(dy, dx) - self.robot_yaw
        goal_angle = math.atan2(math.sin(goal_angle), math.cos(goal_angle))
        sin_angle = math.sin(goal_angle)
        cos_angle = math.cos(goal_angle)
        
        # Apply normalization
        laser_norm = [min(l, self.max_laser_value) / self.max_laser_value for l in laser_ranges]
        dist_norm = min(distance_to_goal, self.max_goal_distance) / self.max_goal_distance
        
        full_observations = laser_norm + [dist_norm, sin_angle, cos_angle]

        return numpy.array(full_observations, dtype=numpy.float32)
    
    def _is_done(self, observations):
        if self._is_failed():
            return True
        if self._is_succeded():
            if self.inference_mode:
                # In inference end at third episode reached
                self.goals_reached_count += 1
                rospy.loginfo(
                    "[INFERENCE] Goal reached (%d/%d)",
                    self.goals_reached_count,
                    self.inference_goal_target_count,
                )

                # End successfully when the target goal count is reached
                if self.goals_reached_count >= self.inference_goal_target_count:
                    rospy.loginfo("[INFERENCE] Episode success: target goals reached")
                    return True

                # Ensure the third goal is always in front of the desk (rectangle)
                if self.goals_reached_count == (self.inference_goal_target_count - 1):
                    self.goal_x = self.inference_third_goal_x
                    self.goal_y = self.inference_third_goal_y
                    rospy.loginfo(
                        "[INFERENCE] Placing final goal at desk front: (%.2f, %.2f)",
                        self.goal_x,
                        self.goal_y,
                    )
                else:
                    self._move_goal_marker()

                self._position_goal_marker()
                self._update_goal_distance_reference()
                self.succeed = False
                return False
            return True
        return False
        
    def _is_failed(self):
        """
        Check if episode should fail due to:
        1. High acceleration (crash impact)
        2. Too close to obstacle (collision)
        3. Maximum steps exceeded
        """

        # Reuse laser data cached in _get_obs() for this step.
        if self._last_min_laser_value is None:
            rospy.logwarn("No valid laser readings!")
            return False
        min_laser_value = self._last_min_laser_value
        
        rospy.logdebug("Min laser distance: %.3f (collision threshold: %.3f)" % (min_laser_value, self.min_range))
        
        if min_laser_value < self.min_range:
            rospy.logerr("COLLISION! Min laser distance: %.3f < %.3f" % (min_laser_value, self.min_range))
            self.fail = True
            return True
        
        if self.current_episode_step >= self.max_episode_steps:
            rospy.logwarn("Max episode steps reached: %d" % self.current_episode_step)
            self.fail = True
            return True

        return False
    
    def _is_succeded(self):
        """Check if robot has reached the goal"""
        dx = self.goal_x - self.robot_x
        dy = self.goal_y - self.robot_y
        distance_to_goal = math.sqrt(dx**2 + dy**2)
        rospy.logwarn("Robot position (%.2f, %.2f), Goal (%.2f, %.2f), Distance to goal: %.3f" % (self.robot_x, self.robot_y, self.goal_x, self.goal_y, distance_to_goal))
        
        if distance_to_goal < self.success_threshold:
            self.succeed = True
            rospy.loginfo("Goal reached! Distance: %.3f meters" % distance_to_goal)
        
        return self.succeed
    
    def _compute_directional_weights(self, relative_angles, max_weight=10.0):
        """Compute normalized angular weights that prioritize frontal obstacles.
        Obstacles near heading 0 rad receive higher weight than side obstacles.
        """
        # Higher power sharpens emphasis around 0 rad.
        power = 6
        raw_weights = (numpy.cos(relative_angles))**power + 0.1
        # Scale then normalize so weights are comparable across scan densities.
        scaled_weights = raw_weights * (max_weight / numpy.max(raw_weights))
        normalized_weights = scaled_weights / numpy.sum(scaled_weights)
        return normalized_weights
    
    def _compute_weighted_obstacle_reward(self, front_ranges, front_angles):
        """Compute obstacle penalty using angle-aware weighting and distance decay.
        Closer and more frontal obstacles produce stronger negative reward.
        """
        if not front_ranges or not front_angles:
            return 0.0

        front_ranges = numpy.array(front_ranges)
        front_angles = numpy.array(front_angles)

        # Only consider obstacles within a local danger radius.
        valid_mask = front_ranges <= 0.5
        if not numpy.any(valid_mask):
            return 0.0

        front_ranges = front_ranges[valid_mask]
        front_angles = front_angles[valid_mask]

        relative_angles = numpy.unwrap(front_angles)
        relative_angles[relative_angles > numpy.pi] -= 2 * numpy.pi

        # Frontal obstacles contribute more than lateral ones.
        weights = self._compute_directional_weights(relative_angles, max_weight=10.0)

        # Convert distances to a smooth risk term: very close -> near 1, far -> near 0.
        safe_dists = numpy.clip(front_ranges - 0.25, 1e-2, 3.5)
        decay = numpy.exp(-3.0 * safe_dists)

        # Weighted aggregate risk from all nearby obstacle rays.
        weighted_decay = numpy.dot(weights, decay)

        # Base penalty with extra scaling by weighted proximity risk.
        reward = - (1.0 + 4.0 * weighted_decay)

        return reward
    
    def _compute_laser_scans(self, observations):
        """
        Computes laser scans and performs min-pooling:
        - considers only 180 laser scans pointing in front of the robot.
        - chunks in 24 groups and takes the minimum, more significant, value
        Reduces input state dimension and training convergence
        """
        target_ray_count = self.new_ranges
        
        num_of_lidar_rays = len(observations.ranges)
        angle_min = observations.angle_min
        angle_increment = observations.angle_increment

        raw_front_ranges = []
        raw_front_angles = []
        
        for i in range(num_of_lidar_rays):
            angle = angle_min + i * angle_increment
            
            # Normalize angle to [0, 2pi)
            if angle < 0:
                angle += 2 * math.pi
            
            # Check if in front sector (Front 180 degrees approx)
            if (0 <= angle <= math.pi/2) or (3*math.pi/2 <= angle <= 2*math.pi):
                dist = observations.ranges[i]
                if numpy.isinf(dist):
                    dist = self.max_laser_value
                elif numpy.isnan(dist):
                    dist = self.min_laser_value
                
                raw_front_ranges.append(dist)
                raw_front_angles.append(angle)

        # if incorrect laser readings
        if len(raw_front_ranges) < target_ray_count:
            return [self.max_laser_value] * target_ray_count, [0.0] * target_ray_count

        # Min-Pooling
        chunk_size = int(len(raw_front_ranges) / target_ray_count)
        
        final_ranges = []
        final_angles = []
        
        for i in range(target_ray_count):
            start_idx = i * chunk_size
            if i == target_ray_count - 1:
                end_idx = len(raw_front_ranges)
            else:
                end_idx = (i + 1) * chunk_size
            
            sector_ranges = raw_front_ranges[start_idx:end_idx]
            sector_angles = raw_front_angles[start_idx:end_idx]
            
            if len(sector_ranges) > 0:
                min_val = min(sector_ranges)
                final_ranges.append(min_val)
                min_idx = sector_ranges.index(min_val)
                final_angles.append(sector_angles[min_idx])
            else:
                final_ranges.append(self.max_laser_value)
                final_angles.append(0.0) 
                
        return raw_front_ranges, raw_front_angles
    
    def _compute_distance_reward(self, current_distance):
        """ Gives positive reward if moving towards goal, negative if moving away """
        if self.previous_distance_to_goal is not None:
            distance_delta = self.previous_distance_to_goal - current_distance
            distance_reward = distance_delta * self.distance_reward_multiplier
            # Clip to prevent extreme values
            distance_reward = numpy.clip(distance_reward, -10.0, 10.0)
        else:
            distance_reward = 0.0
            
        return distance_reward
    
    def _compute_reward(self, observations, done):
        """Computes cumulated reward"""
        dx = self.goal_x - self.robot_x
        dy = self.goal_y - self.robot_y
        current_distance = math.sqrt(dx**2 + dy**2)
        
        goal_angle = math.atan2(dy, dx) - self.robot_yaw
        # Normalize angle to [-pi, pi]
        goal_angle = math.atan2(math.sin(goal_angle), math.cos(goal_angle))

        if self.succeed:
            self.previous_distance_to_goal = current_distance
            rospy.loginfo("SUCCESS REWARD: %.1f" % self.goal_reached_reward)
            return self.goal_reached_reward
        elif self.fail:
            self.previous_distance_to_goal = current_distance
            rospy.loginfo("FAILURE PENALTY: %.1f" % self.obstacle_hit_penalty)
            return self.obstacle_hit_penalty

        distance_reward = self._compute_distance_reward(current_distance)
        self.previous_distance_to_goal = current_distance
    
        # Heading alignment: cos(angle) gives +1 facing goal, -1 facing away
        heading_reward = math.cos(goal_angle) * self.yaw_reward_multiplier

        # Reuse front-sector scan data cached in _get_obs() for this step.
        obstacle_penalty = self._compute_weighted_obstacle_reward(
            self._last_front_ranges,
            self._last_front_angles,
        )
        
        # Reduce penalty near goal
        if current_distance < self.courage_zone_threshold:
            penalty_scale = max(0.2, current_distance / self.courage_zone_threshold)
            obstacle_penalty *= penalty_scale

        # penalize angular velocity magnitude
        turn_penalty = -self.turn_penalty_multiplier * abs(self.angular_speed)
        
        time_pen = self.time_penalty
        
        reward = distance_reward + heading_reward + obstacle_penalty + turn_penalty + time_pen
        
        rospy.logdebug("Reward: dist=%.2f head=%.2f obs=%.2f turn=%.2f time=%.2f total=%.2f" %
                       (distance_reward, heading_reward, obstacle_penalty, turn_penalty, time_pen, reward))
        
        return reward


    # Internal TaskEnv Methods
    def _update_robot_position(self):
        """
        Update current robot position from odometry data
        """
        odom = self.get_odom()
        self.robot_x = odom.pose.pose.position.x
        self.robot_y = odom.pose.pose.position.y
        
        # Extract yaw from quaternion
        orientation_q = odom.pose.pose.orientation
        siny_cosp = 2 * (orientation_q.w * orientation_q.z + orientation_q.x * orientation_q.y)
        cosy_cosp = 1 - 2 * (orientation_q.y * orientation_q.y + orientation_q.z * orientation_q.z)
        self.robot_yaw = math.atan2(siny_cosp, cosy_cosp)

    def get_vector_magnitude(self, vector):
        """
        It calculated the magnitude of the Vector3 given.
        This is usefull for reading imu accelerations and knowing if there has been
        a crash
        :return:
        """
        contact_force_np = numpy.array((vector.x, vector.y, vector.z))
        force_magnitude = numpy.linalg.norm(contact_force_np)

        return force_magnitude
