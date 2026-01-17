# to do

- Refactor start_deepqlearning.py, remove unnecessary implementations and rewards.
- Setup hyperparameters and DQN architectures as state of the art, clean up training loop.
- Refactor reward function calculation and remove curriculum learning from there.
    - assign reward when turtlebot reaches goal
    - assign reward based on distance from goal
    - assign reward based on angle with respect to the goal, meaning that if the turtlebot points toward the goal it gets rewarded
    - assign negative penalty to prevent sway
- Implement curriculum learning by changing environment in stages:
    - first stage has no obstacles
    - then static obstacles are places
    - then obstacles become moving
    - then environment arena becomes a maze with a mix of walls and moving obstacles
    - we setup a training launch file for each training stage and run one after the other as good performances are reached in the previous. If it is possible to change the gazebo environment dynamically then we change the environment with scripts
- Implement monitoring of the performances with live graphs to show rewards
