import rospy
import numpy
from gym import spaces
from openai_ros.robot_envs import turtlebot3_env
from gym.envs.registration import register
from geometry_msgs.msg import Vector3
from openai_ros.task_envs.task_commons import LoadYamlFileParamsTest
from openai_ros.openai_ros_common import ROSLauncher
import os


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
                    launch_file_name="start_world.launch",
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

        # Rewards
        self.forwards_reward = rospy.get_param("/turtlebot3/forwards_reward")
        self.turn_reward = rospy.get_param("/turtlebot3/turn_reward")
        self.end_episode_points = rospy.get_param("/turtlebot3/end_episode_points")

        # Curriculum Learning Parameters
        self.curriculum_stage = rospy.get_param("/turtlebot3/curriculum_stage", 0)
        self.velocity_reward_weight = rospy.get_param("/turtlebot3/velocity_reward_weight", 0.0)
        self.stopping_penalty = rospy.get_param("/turtlebot3/stopping_penalty", 0.0)
        self.distance_reward_weight = rospy.get_param("/turtlebot3/distance_reward_weight", 0.0)
        self.safe_distance_threshold = rospy.get_param("/turtlebot3/safe_distance_threshold", 1.0)
        self.min_velocity_threshold = rospy.get_param("/turtlebot3/min_velocity_threshold", 0.05)
        
        # New Rewards and Threhsolds
        self.step_milestone_reward = rospy.get_param("/turtlebot3/step_milestone_reward", 100.0)
        self.survival_reward = rospy.get_param("/turtlebot3/survival_reward", 200.0)
        self.distance_milestone_reward = rospy.get_param("/turtlebot3/distance_milestone_reward", 10.0)
        self.distance_milestone_interval = rospy.get_param("/turtlebot3/distance_milestone_interval", 1.0)
        
        # Stages Limits
        self.stage_0_min_steps = rospy.get_param("/turtlebot3/stage_0_min_steps", 300)
        self.stage_0_max_steps = rospy.get_param("/turtlebot3/stage_0_max_steps", 500)
        
        self.stage_1_min_steps = rospy.get_param("/turtlebot3/stage_1_min_steps", 500)
        self.stage_1_max_steps = rospy.get_param("/turtlebot3/stage_1_max_steps", 700)
        
        self.stage_2_min_steps = rospy.get_param("/turtlebot3/stage_2_min_steps", 700)
        self.stage_2_max_steps = rospy.get_param("/turtlebot3/stage_2_max_steps", 1000)

        self.cumulated_steps = 0.0
        self.previous_position = None
        self.current_velocity = 0.0
        
        # Current Stage Limits
        self.current_min_steps = self.stage_0_min_steps
        self.current_max_steps = self.stage_0_max_steps
        
        # Initialize reward details for info
        self.reward_props = {}

        rospy.logwarn("="*50)
        rospy.logwarn("CURRICULUM LEARNING INITIALIZED")
        rospy.logwarn("Current Stage: " + str(self.curriculum_stage))
        rospy.logwarn("Stage 0: Basic collision avoidance")
        rospy.logwarn("Stage 1: Fast continuous movement")
        rospy.logwarn("Stage 2: Maximize distance from obstacles")
        rospy.logwarn("="*50)


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
        # For Info Purposes
        self.cumulated_reward = 0.0
        # Set to false Done, because its calculated asyncronously
        self._episode_done = False
        
        # Reset velocity tracking for curriculum learning
        odom = self.get_odom()
        self.previous_position = odom.pose.pose.position
        self.initial_position = odom.pose.pose.position
        self.current_velocity = 0.0
        
        # New Envs
        self.episode_steps = 0
        self.max_distance_milestone_reached = 0.0
        self.reward_props = {
            'base': 0.0,
            'velocity': 0.0,
            'distance_clearance': 0.0,
            'step_milestone': 0.0,
            'survival': 0.0,
            'distance_traveled_milestone': 0.0
        }


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

        rospy.logdebug("END Set Action ==>"+str(action))

    def _get_obs(self):
        """
        Here we define what sensor data defines our robots observations
        To know which Variables we have acces to, we need to read the
        TurtleBot2Env API DOCS
        :return:
        """
        rospy.logdebug("Start Get Observation ==>")
        # We get the laser scan data
        laser_scan = self.get_laser_scan()

        discretized_observations = self.discretize_scan_observation(    laser_scan,
                                                                        self.new_ranges
                                                                        )

        rospy.logdebug("Observations==>"+str(discretized_observations))
        rospy.logdebug("END Get Observation ==>")
        return discretized_observations


    def _is_done(self, observations):

        if self._episode_done:
            rospy.logerr("TurtleBot2 is Too Close to wall==>")
        else:
            rospy.logwarn("TurtleBot2 is NOT close to a wall ==>")

        # Now we check if it has crashed based on the imu
        imu_data = self.get_imu()
        linear_acceleration_magnitude = self.get_vector_magnitude(imu_data.linear_acceleration)
        if linear_acceleration_magnitude > self.max_linear_aceleration:
            rospy.logerr("TurtleBot2 Crashed==>"+str(linear_acceleration_magnitude)+">"+str(self.max_linear_aceleration))
            self._episode_done = True
        else:
            rospy.logerr("DIDNT crash TurtleBot2 ==>"+str(linear_acceleration_magnitude)+">"+str(self.max_linear_aceleration))

        # Check for Max Steps (Survival)
        if self.episode_steps >= self.current_max_steps:
            rospy.logwarn("TurtleBot3 Survived Max Steps ==> " + str(self.episode_steps))
            self._episode_done = True

        return self._episode_done

    def step(self, action):
        obs, reward, done, info = super(TurtleBot3WorldEnv, self).step(action)
        info.update(self.reward_props)
        return obs, reward, done, info

    def _compute_reward(self, observations, done):
        """
        Compute reward based on curriculum stage and milestones.
        """
        
        # Initialize breakdown for this step
        step_reward_props = {
            'base': 0.0,
            'velocity': 0.0,
            'distance_clearance': 0.0,
            'step_milestone': 0.0,
            'survival': 0.0,
            'distance_traveled_milestone': 0.0
        }
        
        self.episode_steps += 1
        
        # Check if we survived (done checking happens in _is_done, so if done=True and not crashed)
        # We need to distinguish crash vs survival
        # If done is True, we check if we reached max steps
        is_survival = (self.episode_steps >= self.current_max_steps)
        
        if not done:
            # Base reward for action
            if self.last_action == "FORWARDS":
                step_reward_props['base'] = self.forwards_reward
            else:
                step_reward_props['base'] = self.turn_reward
            
            # Stage 0: Basic collision avoidance
            if self.curriculum_stage == 0:
                pass # Only base reward + milestones
            
            # Stage 1: Add velocity-based rewards
            elif self.curriculum_stage == 1:
                velocity_reward = self._compute_velocity_reward()
                step_reward_props['velocity'] = (self.velocity_reward_weight * velocity_reward)
            
            # Stage 2: Add distance clearance rewards
            elif self.curriculum_stage == 2:
                velocity_reward = self._compute_velocity_reward()
                distance_reward = self._compute_distance_reward(observations)
                step_reward_props['velocity'] = (self.velocity_reward_weight * velocity_reward)
                step_reward_props['distance_clearance'] = (self.distance_reward_weight * distance_reward)

            # --- Step Milestones ---
            if self.episode_steps == self.current_min_steps:
                 rospy.logwarn(">>> REACHED MIN STEP THRESHOLD " + str(self.current_min_steps) + "! REWARD BONUS!")
                 step_reward_props['step_milestone'] = self.step_milestone_reward

            # --- Distance Travelled Milestones ---
            odom = self.get_odom()
            current_pos = odom.pose.pose.position
            dx = current_pos.x - self.initial_position.x
            dy = current_pos.y - self.initial_position.y
            dist_from_start = numpy.sqrt(dx*dx + dy*dy)
            
            if dist_from_start > (self.max_distance_milestone_reached + self.distance_milestone_interval):
                 rospy.logwarn(">>> REACHED DISTANCE MILESTONE " + str(dist_from_start) + "m!")
                 step_reward_props['distance_traveled_milestone'] = self.distance_milestone_reward
                 self.max_distance_milestone_reached += self.distance_milestone_interval # Advance milestone

        else:
            if is_survival:
                rospy.logwarn(">>> SURVIVAL REWARD GRANTED!")
                step_reward_props['survival'] = self.survival_reward
            else:
                # Crash
                step_reward_props['base'] = -1*self.end_episode_points

        # Sum up
        reward = sum(step_reward_props.values())

        rospy.logdebug("reward_details=" + str(step_reward_props))
        self.reward_props = step_reward_props
        
        self.cumulated_reward += reward
        self.cumulated_steps += 1

        return reward


    def _compute_velocity_reward(self):
        """
        Compute reward based on robot velocity.
        Rewards forward movement and penalizes stopping or slow movement.
        """
        # Get current odometry
        odom = self.get_odom()
        current_pos = odom.pose.pose.position
        
        # Calculate velocity based on position change
        if self.previous_position is not None:
            dx = current_pos.x - self.previous_position.x
            dy = current_pos.y - self.previous_position.y
            distance_moved = numpy.sqrt(dx*dx + dy*dy)
            # Approximate velocity (distance per step)
            self.current_velocity = distance_moved
        
        self.previous_position = current_pos
        
        # Reward faster movement, penalize stopping
        if self.current_velocity < self.min_velocity_threshold:
            # Penalty for stopping or very slow movement (-5)
            velocity_reward = -self.stopping_penalty
        else:
            # Reward proportional to velocity
            velocity_reward = self.current_velocity * 15.0
        
        # Cap reward to prevent exploitation
        velocity_reward = numpy.clip(velocity_reward, -10.0, 30.0)
        
        return velocity_reward
    
    def _compute_distance_reward(self, observations):
        """
        Compute reward based on distance from obstacles.
        Rewards maintaining safe distance from walls while navigating.
        """
        # Convert observations to numpy array
        distances = numpy.array(observations)
        
        # Calculate minimum distance to any obstacle
        min_distance = numpy.min(distances)
        
        # Calculate average distance (general clearance)
        avg_distance = numpy.mean(distances)
        
        # Reward for maintaining safe distance
        if min_distance < self.safe_distance_threshold:
            # Penalty for being too close (but not crashing)
            distance_reward = -2.0 * (self.safe_distance_threshold - min_distance)
        else:
            # Reward for maintaining good clearance
            distance_reward = 2.0 * min_distance + 1.5 * avg_distance
        
        # Cap the reward to prevent exploitation
        distance_reward = numpy.clip(distance_reward, -10.0, 30.0)
        
        return distance_reward
    
    def update_curriculum_stage(self, new_stage, new_velocity_weight=None, new_distance_weight=None):
        """
        Update the curriculum learning stage.
        Called from training script when performance threshold is met.
        """
        self.curriculum_stage = new_stage
        
        if new_velocity_weight is not None:
            self.velocity_reward_weight = new_velocity_weight
        
        if new_distance_weight is not None:
            self.distance_reward_weight = new_distance_weight
            
        # Update Step Thresholds
        if self.curriculum_stage == 0:
            self.current_min_steps = self.stage_0_min_steps
            self.current_max_steps = self.stage_0_max_steps
        elif self.curriculum_stage == 1:
            self.current_min_steps = self.stage_1_min_steps
            self.current_max_steps = self.stage_1_max_steps
        elif self.curriculum_stage >= 2:
            self.current_min_steps = self.stage_2_min_steps
            self.current_max_steps = self.stage_2_max_steps
        
        rospy.logwarn("="*50)
        rospy.logwarn("CURRICULUM STAGE UPDATED!")
        rospy.logwarn("New Stage: " + str(self.curriculum_stage))
        rospy.logwarn("Min Steps: " + str(self.current_min_steps))
        rospy.logwarn("Max Steps: " + str(self.current_max_steps))
        rospy.logwarn("Velocity Reward Weight: " + str(self.velocity_reward_weight))
        rospy.logwarn("Distance Reward Weight: " + str(self.distance_reward_weight))
        if self.curriculum_stage == 0:
            rospy.logwarn("Focus: Basic collision avoidance + Survival 300-500")
        elif self.curriculum_stage == 1:
            rospy.logwarn("Focus: Fast continuous movement + Survival 500-700")
        elif self.curriculum_stage == 2:
            rospy.logwarn("Focus: Maximize distance from obstacles + Survival 700-1000")
        rospy.logwarn("="*50)

    # Internal TaskEnv Methods

    def discretize_scan_observation(self,data,new_ranges):
        """
        Discards all the laser readings that are not multiple in index of new_ranges
        value.
        """
        self._episode_done = False

        discretized_ranges = []
        mod = len(data.ranges)/new_ranges

        rospy.logdebug("data=" + str(data))
        rospy.logdebug("new_ranges=" + str(new_ranges))
        rospy.logdebug("mod=" + str(mod))

        for i, item in enumerate(data.ranges):
            if (i%mod==0):
                if item == float ('Inf') or numpy.isinf(item):
                    discretized_ranges.append(self.max_laser_value)
                elif numpy.isnan(item):
                    discretized_ranges.append(self.min_laser_value)
                else:
                    discretized_ranges.append(int(item))

                if (self.min_range > item > 0):
                   # rospy.logerr("done Validation >>> item=" + str(item)+"< "+str(self.min_range))
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

