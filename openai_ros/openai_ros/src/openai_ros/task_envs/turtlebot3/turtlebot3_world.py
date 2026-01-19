import rospy
import numpy
from gym import spaces
from openai_ros.robot_envs import turtlebot3_env
from gym.envs.registration import register
from geometry_msgs.msg import Vector3, Pose
from openai_ros.task_envs.task_commons import LoadYamlFileParamsTest
from openai_ros.openai_ros_common import ROSLauncher
from gazebo_msgs.srv import SpawnModel, DeleteModel
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
        
        # Goal position - randomized each episode
        self.goal_x = random.uniform(-2.0, 2.0)
        self.goal_y = random.uniform(-2.0, 2.0)

        # We create two arrays based on the binary values that will be assigned
        # In the discretization method.
        laser_scan = self.get_laser_scan()
        num_laser_readings = int(len(laser_scan.ranges)/self.new_ranges)
        high = numpy.full((num_laser_readings), self.max_laser_value)
        low = numpy.full((num_laser_readings), self.min_laser_value)

        # We only use two integers
        self.observation_space = spaces.Box(low, high)

        rospy.logdebug("ACTION SPACES TYPE===>"+str(self.action_space))
        rospy.logdebug("OBSERVATION SPACES TYPE===>"+str(self.observation_space))

        self.cumulated_steps = 0.0
        
        # Initialize robot position tracking
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        self.previous_distance_to_goal = None
        
        # Wait for Gazebo services to spawn goal marker
        rospy.loginfo("Waiting for Gazebo model services...")
        rospy.wait_for_service('/gazebo/spawn_sdf_model')
        rospy.wait_for_service('/gazebo/delete_model')
        self.spawn_model_srv = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        self.delete_model_srv = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        rospy.loginfo("Gazebo services ready")
        
        # Spawn initial goal marker
        self._spawn_goal_marker()

    def _spawn_goal_marker(self):
        """Spawn or update goal marker in Gazebo as an actual model"""
        # Delete old marker if it exists
        try:
            self.delete_model_srv('goal_marker')
            rospy.sleep(0.1)
        except:
            pass  # Model didn't exist yet
        
        # SDF model definition for a green cylinder
        goal_sdf = """<?xml version='1.0'?>
        <sdf version='1.6'>
          <model name='goal_marker'>
            <static>true</static>
            <link name='link'>
              <visual name='visual'>
                <geometry>
                  <cylinder>
                    <radius>0.25</radius>
                    <length>0.2</length>
                  </cylinder>
                </geometry>
                <material>
                  <ambient>0 1 0 0.8</ambient>
                  <diffuse>0 1 0 0.8</diffuse>
                  <specular>0 1 0 1</specular>
                  <emissive>0 0.5 0 1</emissive>
                </material>
              </visual>
            </link>
          </model>
        </sdf>"""
        
        # Set pose for the goal marker
        goal_pose = Pose()
        goal_pose.position.x = self.goal_x
        goal_pose.position.y = self.goal_y
        goal_pose.position.z = 0.1
        goal_pose.orientation.w = 1.0
        
        # Spawn the model in Gazebo
        try:
            self.spawn_model_srv('goal_marker', goal_sdf, '', goal_pose, 'world')
            rospy.loginfo("Goal marker spawned in Gazebo at (%.2f, %.2f)" % (self.goal_x, self.goal_y))
        except Exception as e:
            rospy.logerr("Failed to spawn goal marker: %s" % str(e))


    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        self.move_base( self.init_linear_forward_speed,
                        self.init_linear_turn_speed,
                        epsilon=0.05,
                        update_rate=10)

        return True


    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        of an episode.
        :return:
        """
        
        # Randomize new goal position for this episode
        self.goal_x = random.uniform(-4.0, 4.0)
        self.goal_y = random.uniform(-4.0, 4.0)
        
        # Update robot position from odometry
        self._update_robot_position()
        
        # Calculate initial distance to goal
        dx = self.goal_x - self.robot_x
        dy = self.goal_y - self.robot_y
        self.previous_distance_to_goal = math.sqrt(dx**2 + dy**2)
        
        self.succeed = False
        self.fail = False
        
        # Reset episode step counter
        self.current_episode_step = 0
        
        # Spawn the new goal marker in Gazebo
        self._spawn_goal_marker()
        rospy.loginfo("New goal set at: (%.2f, %.2f), initial distance: %.2f" % (self.goal_x, self.goal_y, self.previous_distance_to_goal))


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
        """
        Here we define what sensor data defines our robots observations
        To know which Variables we have acces to, we need to read the
        TurtleBot2Env API DOCS
        :return:
        """
        rospy.logdebug("Start Get Observation ==>")
        
        # Update robot position
        self._update_robot_position()
        
        # We get the laser scan data
        laser_scan = self.get_laser_scan()

        discretized_observations = self.discretize_scan_observation(    laser_scan,
                                                                        self.new_ranges
                                                                        )

        rospy.logdebug("Observations==>"+str(discretized_observations))
        rospy.logdebug("Robot position: (%.2f, %.2f, %.2f), Goal: (%.2f, %.2f)" % 
                      (self.robot_x, self.robot_y, self.robot_yaw, self.goal_x, self.goal_y))
        rospy.logdebug("END Get Observation ==>")
        return discretized_observations
    
    def _is_done(self, observations):
        
        if self._is_failed(observations) or self._is_succeded(observations):
            return True
        else:
            return False

    def _is_failed(self, observations):
        """
        Check if episode should fail due to:
        1. High acceleration (crash impact)
        2. Too close to obstacle (collision)
        3. Maximum steps exceeded
        """
        # Check IMU for crash detection
        imu_data = self.get_imu()
        linear_acceleration_magnitude = self.get_vector_magnitude(imu_data.linear_acceleration)
        if linear_acceleration_magnitude > self.max_linear_aceleration:
            rospy.logerr("CRASH DETECTED! Acceleration: %.2f > %.2f" % (linear_acceleration_magnitude, self.max_linear_aceleration))
            self.fail = True
            return True

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
    
    def _is_succeded(self, observations):
        """Check if robot has reached the goal"""
        dx = self.goal_x - self.robot_x
        dy = self.goal_y - self.robot_y
        distance_to_goal = math.sqrt(dx**2 + dy**2)
        
        if distance_to_goal < 0.2:
            self.succeed = True
            rospy.loginfo("Goal reached! Distance: %.3f meters" % distance_to_goal)
        
        return self.succeed
    
    def _compute_directional_weights(self, relative_angles, max_weight=10.0):
        power = 6
        raw_weights = (numpy.cos(relative_angles))**power + 0.1
        scaled_weights = raw_weights * (max_weight / numpy.max(raw_weights))
        normalized_weights = scaled_weights / numpy.sum(scaled_weights)
        return normalized_weights
    
    def _compute_obstacle_penalty(self, laser_ranges):
        """
        Compute weighted penalty based on obstacle proximity and direction.
        Obstacles directly in front are penalized more heavily.
        """
        # Convert laser ranges to numpy array
        ranges = numpy.array(laser_ranges)
        
        # Calculate angle for each laser reading
        num_readings = len(ranges)
        angles = numpy.linspace(-numpy.pi, numpy.pi, num_readings)
        
        # Only consider obstacles within danger zone (0.5m)
        valid_mask = (ranges <= 0.5) & (ranges > 0)
        if not numpy.any(valid_mask):
            return 0.0  # No close obstacles
        
        # Filter to only dangerous obstacles
        danger_ranges = ranges[valid_mask]
        danger_angles = angles[valid_mask]
        
        # Normalize angles to [-pi, pi]
        danger_angles = numpy.arctan2(numpy.sin(danger_angles), numpy.cos(danger_angles))
        
        # Calculate directional weights (front obstacles matter more)
        weights = self._compute_directional_weights(danger_angles, max_weight=10.0)
        
        # Calculate distance-based penalty (closer = worse)
        safe_distances = numpy.clip(danger_ranges - 0.2, 1e-2, 3.5)  # 0.2m safety margin
        decay = numpy.exp(-3.0 * safe_distances)
        
        # Weighted combination
        weighted_decay = numpy.dot(weights, decay)
        penalty = -(1.0 + 4.0 * weighted_decay)
        
        return penalty

    def _compute_reward(self, observations, done):
        """
        Compute reward based on:
        1. Progress toward goal (yaw alignment)
        2. Obstacle avoidance (weighted by direction)
        3. Terminal states (success/failure)
        """
        # Calculate goal alignment reward
        dx = self.goal_x - self.robot_x
        dy = self.goal_y - self.robot_y
        goal_angle = math.atan2(dy, dx) - self.robot_yaw
        # Normalize angle to [-pi, pi]
        goal_angle = math.atan2(math.sin(goal_angle), math.cos(goal_angle))
        
        yaw_reward = 1.0 - (2.0 * abs(goal_angle) / math.pi)
        
        # Calculate obstacle penalty (using raw laser data from observations)
        obstacle_penalty = self._compute_obstacle_penalty(observations)
        
        # Combine rewards
        reward = yaw_reward + obstacle_penalty
        
        # Terminal rewards override
        if self.succeed:
            reward = 200.0
            rospy.loginfo("SUCCESS! Goal reached!")
        elif self.fail:
            reward = -100.0
            rospy.logerr("FAILURE! Collision detected!")

        rospy.logdebug("yaw_reward=%.2f, obstacle_penalty=%.2f, total_reward=%.2f" % (yaw_reward, obstacle_penalty, reward))
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

    def discretize_scan_observation(self,data,new_ranges):
        """
        Discards all the laser readings that are not multiple in index of new_ranges
        value.
        """
        self._episode_done = False

        discretized_ranges = []
        mod = len(data.ranges)/new_ranges
        
        for i, item in enumerate(data.ranges):
            if (i%mod==0):
                if item == float ('Inf') or numpy.isinf(item):
                    discretized_ranges.append(self.max_laser_value)
                elif numpy.isnan(item):
                    discretized_ranges.append(self.min_laser_value)
                else:
                    discretized_ranges.append(int(item))

                if (self.min_range > item > 0):
                    rospy.logerr("done Validation >>> item=" + str(item)+"< "+str(self.min_range))
                    self._episode_done = True
                else:
                    rospy.logdebug("NOT done Validation >>> item=" + str(item)+"< "+str(self.min_range))


        return discretized_ranges


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
