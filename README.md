## DQN Reinforcement Learning for Robot navigation

This project aims to train a DQN model to navigate a Turtlebot3 in environments simulated in the Gazebo robotics simulator.

To be able to interface with Gazebo and look at the simulation while training, start the Xserver and that connects acts as a proxy between the host and the container.

Start the container running the following command:

```bash
docker compose up -d
```

Enter the container in interactive mode:

```bash
docker exec -it final_project bash
```

After building for the first time compile ROS using catkin:

```bash
catkin build
source devel/setup.bash
```

To run a training or inference execute the corresponding launch file, e.g.:

```bash
roslaunch curriculum_learning start_training_stage1.launch
```

If running the project from a Linux distribution, ensure that Docker has the Nvidia toolkit installed and correctly configured.
Ensure the container has authorizations to communicate with the Xserver, run this command:

```bash
xhost +local:docker
```

Then run the container by executing

```bash
docker compose -f docker-compose.linux.yml up -d
```

Enter the container in interactive mode:

```bash
docker exec -it final_project_linux bash
```

Run the training with the same ROS commands as above.
