import rospy
import numpy
from gym import spaces
from openai_ros.robot_envs import turtlebot3_env
from gym.envs.registration import register
from geometry_msgs.msg import Vector3, Pose, Quaternion
from openai_ros.task_envs.task_commons import LoadYamlFileParamsTest
from openai_ros.openai_ros_common import ROSLauncher
from gazebo_msgs.srv import SpawnModel, DeleteModel, SetModelState
from gazebo_msgs.msg import ModelState
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

        # Only variable needed to be set here
        number_actions = rospy.get_param('/turtlebot3/n_actions')
        self.action_space = spaces.Discrete(number_actions)

        # We set the reward range, which is not compulsory but here we do it.
        self.reward_range = (-numpy.inf, numpy.inf)


        #number_observations = rospy.get_param('/turtlebot3/n_observations')
        """
        We set the Observation space for the 6 observations
        cube_observations = [
            round(current_disk_roll_vel, 0),
            round(y_distance, 1),
            round(roll, 1),
            round(pitch, 1),
            round(y_linear_speed,1),
            round(yaw, 1),
        ]
        """

        # Actions and Observations
        self.linear_forward_speed = rospy.get_param('/turtlebot3/linear_forward_speed')
        self.linear_turn_speed = rospy.get_param('/turtlebot3/linear_turn_speed')
        self.angular_speed = rospy.get_param('/turtlebot3/angular_speed')
        self.init_linear_forward_speed = rospy.get_param('/turtlebot3/init_linear_forward_speed')
        self.init_linear_turn_speed = rospy.get_param('/turtlebot3/init_linear_turn_speed')

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

        # # We create two arrays based on the binary values that will be assigned
        # # In the discretization method.
        laser_scan = self.get_laser_scan()
        
        total_laser_readings = len(laser_scan.ranges)
        num_laser_readings = int(total_laser_readings / self.new_ranges)
        
        rospy.loginfo(f"Laser readings: {total_laser_readings} total, sampling every {self.new_ranges}th = {num_laser_readings} readings")
        laser_ranges, _ = self._compute_laser_scans(laser_scan)
        num_laser_readings = len(laser_ranges)
        # Calculate max possible distance within arena
        
        self.max_goal_distance = math.sqrt((self.arena_max_x - self.arena_min_x)**2 + 
                                           (self.arena_max_y - self.arena_min_y)**2)
        
        # Observation space: [laser_readings..., distance_to_goal, angle_to_goal]
        laser_high = numpy.full((num_laser_readings,), self.max_laser_value, dtype=numpy.float32)
        laser_low = numpy.full((num_laser_readings,), self.min_laser_value, dtype=numpy.float32)
        obs_high = numpy.concatenate([laser_high, 
                                      numpy.array([self.max_goal_distance, math.pi], dtype=numpy.float32)])
        obs_low = numpy.concatenate([laser_low, 
                                     numpy.array([0.0, -math.pi], dtype=numpy.float32)])
        
        obs_dim = num_laser_readings + 2  # laser + [distance_to_goal, angle_to_goal]
        # Observation space includes goal coordinates
        self.observation_space = spaces.Box(obs_low, obs_high, shape=(obs_dim,), dtype=numpy.float32)

        rospy.logdebug("ACTION SPACES TYPE===>"+str(self.action_space))
        rospy.logdebug("OBSERVATION SPACES TYPE===>"+str(self.observation_space))

        self.cumulated_steps = 0.0
        
        # Initialize robot position tracking
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        self.previous_distance_to_goal = None
        
        # Wait for Gazebo service to move goal marker
        rospy.loginfo("Waiting for Gazebo set_model_state service...")
        rospy.wait_for_service('/gazebo/set_model_state')
        self.set_model_state_srv = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        rospy.loginfo("Gazebo service ready")
        
        self._move_goal_marker()
        

    def _move_goal_marker(self):
        """Move the existing goal marker to a new position using SetModelState"""
        angle = random.uniform(0, 2 * math.pi)  # Random direction (0 to 360 degrees)
        distance = 1.0  # Fixed 1 meter distance in any direction
        
        # Calculate goal position using polar coordinates
        self.goal_x = self.robot_x + distance * math.cos(angle)
        self.goal_y = self.robot_y + distance * math.sin(angle)
        
        # Clamp to safe arena boundaries
        self.goal_x = numpy.clip(self.goal_x, self.safe_arena_min_x, self.safe_arena_max_x)
        self.goal_y = numpy.clip(self.goal_y, self.safe_arena_min_y, self.safe_arena_max_y)
        
        # Calculate actual distance to goal (after clamping)
        dx = self.goal_x - self.robot_x
        dy = self.goal_y - self.robot_y
        self.previous_distance_to_goal = math.sqrt(dx**2 + dy**2)
        
    def _set_init_pose(self):
        """Sets the Robot in its init pose"""
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
        
        self._update_robot_position()
        
        # Move the goal marker to new position in Gazebo
        try:
            # Create model state message
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

    def _set_action(self, action):
        """
        This set action will Set the linear and angular speed of the turtlebot2
        based on the action number given.
        :param action: The action integer that set s what movement to do next.
        """

        rospy.logdebug("Start Set Action ==>"+str(action))
        # We convert the actions to speed movements to send to the parent class CubeSingleDiskEnv
        if action == 0: #FORWARD
            linear_speed = self.linear_forward_speed
            angular_speed = 0.0
            self.last_action = "FORWARDS"
        elif action == 1: #LEFT
            linear_speed = self.linear_turn_speed
            angular_speed = self.angular_speed
            self.last_action = "TURN_LEFT"
        elif action == 2: #RIGHT
            linear_speed = self.linear_turn_speed
            angular_speed = -1*self.angular_speed
            self.last_action = "TURN_RIGHT"

        # We tell TurtleBot2 the linear and angular speed to set to execute
        self.move_base(linear_speed, angular_speed, epsilon=0.05, update_rate=10)
        
        # Increment episode step counter
        self.current_episode_step += 1

        rospy.logdebug("END Set Action ==>"+str(action))

    def _get_obs(self):
        rospy.logdebug("Start Get Observation ==>")
        
        self._update_robot_position()
        laser_scan = self.get_laser_scan()

        laser_ranges, _ = self._compute_laser_scans(laser_scan)
        # Calculate relative goal information
        dx = self.goal_x - self.robot_x
        dy = self.goal_y - self.robot_y
        distance_to_goal = math.sqrt(dx**2 + dy**2)
        
        # Calculate angle to goal relative to robot's heading (Normalized to [-pi, pi])
        goal_angle = math.atan2(dy, dx) - self.robot_yaw
        goal_angle = math.atan2(math.sin(goal_angle), math.cos(goal_angle))
        
        # # --- NORMALIZATION FIX ---
        # # Normalize Laser readings [0, max_laser] -> [0, 1]
        # laser_norm = [min(l, self.max_laser_value) / self.max_laser_value for l in ranges]
        
        # # Normalize Distance (Approximate max diagonal of 3x3 arena is ~4.3m)
        # dist_norm = min(distance_to_goal, self.max_goal_distance) / self.max_goal_distance
        
        # # Normalize Angle [-pi, pi] -> [-1, 1]
        # angle_norm = goal_angle / math.pi
        
        # # The Vector: [Laser0, Laser1, ..., LaserN, Distance, Angle]
        # full_observations = laser_norm + [dist_norm, angle_norm]
        
        full_observations = laser_ranges + [distance_to_goal, goal_angle]

        return numpy.array(full_observations, dtype=numpy.float32)
    
    def _is_done(self, observations):
        return self._is_failed()
        
    def _is_failed(self):
        """
        Check if episode should fail due to:
        1. High acceleration (crash impact)
        2. Too close to obstacle (collision)
        3. Maximum steps exceeded
        """
        # Check IMU for crash detection
        # imu_data = self.get_imu()
        # linear_acceleration_magnitude = self.get_vector_magnitude(imu_data.linear_acceleration)
        # if linear_acceleration_magnitude > self.max_linear_aceleration:
        #     rospy.logerr("CRASH DETECTED! Acceleration: %.2f > %.2f" % (linear_acceleration_magnitude, self.max_linear_aceleration))
        #     self.fail = True
        #     return True

        laser_scan = self.get_laser_scan()
        
        # Filter out invalid readings (inf, nan, 0)
        valid_ranges = [r for r in laser_scan.ranges if not (numpy.isinf(r) or numpy.isnan(r) or r == 0.0)]
        
        if len(valid_ranges) == 0:
            rospy.logwarn("No valid laser readings!")
            return False
            
        min_laser_value = min(valid_ranges)
        
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
            # move goal
            self._move_goal_marker()
            rospy.loginfo("New goal at: (%.2f, %.2f), distance: %.2fm" % (self.goal_x, self.goal_y, self.previous_distance_to_goal))
            self.succeed = True
            rospy.loginfo("Goal reached! Distance: %.3f meters" % distance_to_goal)
        
        return self.succeed
    
    def _compute_directional_weights(self, relative_angles, max_weight=10.0):
        power = 6
        raw_weights = (numpy.cos(relative_angles))**power + 0.1
        scaled_weights = raw_weights * (max_weight / numpy.max(raw_weights))
        normalized_weights = scaled_weights / numpy.sum(scaled_weights)
        return normalized_weights
    
    def _compute_weighted_obstacle_reward(self, front_ranges, front_angles):
        if not front_ranges or not front_angles:
            return 0.0

        front_ranges = numpy.array(front_ranges)
        front_angles = numpy.array(front_angles)

        valid_mask = front_ranges <= 0.5
        if not numpy.any(valid_mask):
            return 0.0

        front_ranges = front_ranges[valid_mask]
        front_angles = front_angles[valid_mask]

        relative_angles = numpy.unwrap(front_angles)
        relative_angles[relative_angles > numpy.pi] -= 2 * numpy.pi

        weights = self._compute_directional_weights(relative_angles, max_weight=10.0)

        safe_dists = numpy.clip(front_ranges - 0.25, 1e-2, 3.5)
        decay = numpy.exp(-3.0 * safe_dists)

        weighted_decay = numpy.dot(weights, decay)

        reward = - (1.0 + 4.0 * weighted_decay)

        return reward
    
    def _compute_laser_scans(self, observations):
        scan_ranges = []
        front_ranges = []
        front_angles = []
        num_of_lidar_rays = len(observations.ranges)
        angle_min = observations.angle_min
        angle_increment = observations.angle_increment

        for i in range(num_of_lidar_rays):
            angle = angle_min + i * angle_increment
            distance = observations.ranges[i]

            if distance == float ('Inf') or numpy.isinf(distance):
                scan_ranges.append(self.max_laser_value)
            elif numpy.isnan(distance):
                scan_ranges.append(self.min_laser_value)
            else:
                scan_ranges.append(float(distance))

            if (0 <= angle <= math.pi/2) or (3*math.pi/2 <= angle <= 2*math.pi):
                front_ranges.append(distance)
                front_angles.append(angle)

        return front_ranges, front_angles
        

    def _compute_reward(self, observations, done):
        # 1. Extract pieces from the observation vector
        # observations = [laser_0 ... laser_n, distance, angle]
        # 1. Recalculate raw physical values for accurate reward logic.
        #    We do not use 'observations' directly here because it contains normalized values (0-1),
        #    which breaks physical distance delta calculations.
        dx = self.goal_x - self.robot_x
        dy = self.goal_y - self.robot_y
        current_distance = math.sqrt(dx**2 + dy**2)
        
        goal_angle = math.atan2(dy, dx) - self.robot_yaw
        # Normalize angle to [-pi, pi]
        goal_angle = math.atan2(math.sin(goal_angle), math.cos(goal_angle))

        if self.previous_distance_to_goal is not None:
            distance_delta = self.previous_distance_to_goal - current_distance
            distance_reward = distance_delta * 100.0  # Scale to make meaningful
        else:
            distance_reward = 0.0
        
        self.previous_distance_to_goal = current_distance
    
        # 2. Alignment Reward (from Script 1)
        # 1.0 if facing goal, -1.0 if facing away.
        yaw_reward = 1.0 - (2.0 * abs(goal_angle) / math.pi) * 0.1
        
        # 3. Obstacle Penalty (using our new weighted function)
        front_ranges, front_angles = self._compute_laser_scans(self.laser_scan)
        obstacle_penalty = self._compute_weighted_obstacle_reward(front_ranges, front_angles)
        # Living penalty to encourage faster completion
        #time_penalty = -0.2
    
        # 4. Total step reward
        reward = distance_reward + yaw_reward + obstacle_penalty #+ time_penalty
        
        # 5. Terminal Rewards (Overriding step rewards)
        if self._is_succeded():
            reward = 100.0
            self.succeed = False
        elif self.fail:
            reward = -50.0
        
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
