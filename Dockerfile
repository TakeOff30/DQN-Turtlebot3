FROM osrf/ros:noetic-desktop-full

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install --no-install-recommends -y \
    vim \
    wget \
    doxygen \
    graphviz \
    build-essential \
    libeigen3-dev \
    python3-catkin-tools \
    ros-noetic-dynamic-reconfigure \
    ros-noetic-tf2-ros \
    ros-noetic-navigation \
    x11-apps \
    firefox \
    git \
    unzip \
    python3-pip  \
    python3-tk \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /catkin_build_ws/src \
    && mkdir -p /catkin_make_ws/src

WORKDIR /catkin_build_ws
RUN /bin/bash -c "source /opt/ros/noetic/setup.sh; rosdep init && rosdep update; \
                  source /opt/ros/noetic/setup.sh; catkin init; catkin config --extend /opt/ros/noetic; \
                  source /opt/ros/noetic/setup.sh; catkin build;"


WORKDIR /catkin_make_ws
RUN /bin/bash -c "source /opt/ros/noetic/setup.sh; catkin_make;"

RUN useradd -m  -s /bin/bash ubuntu
RUN usermod -aG sudo ubuntu && echo "ubuntu ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/ubuntu
RUN chmod 044 /etc/sudoers.d/ubuntu
USER ubuntu:ubuntu
WORKDIR /home/ubuntu
RUN echo "source /opt/ros/noetic/setup.bash" >> /home/ubuntu/.bashrc

# create catkin ws (Code is now injected via Volumes, not COPY)
RUN mkdir -p /home/ubuntu/simulation_ws/src
WORKDIR /home/ubuntu/simulation_ws/src

# The folders 'my_turtlebot3_openai_example', 'openai_ros', 'turtlebot3' 
# will be mounted as volumes when running with docker-compose

# install Python dependencies
RUN python3 -m pip install gym==0.25.0
RUN python3 -m pip install GitPython
RUN python3 -m pip install torch
RUN python3 -m pip install python3-pyqt5
RUN python3 -m pip install python3-pyqtgraph
RUN python3 -m pip install std_msgs

WORKDIR /home/ubuntu/simulation_ws
# Initialize the workspace structure. You must run 'catkin build' manually after mounting volumes.
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash; catkin init"
RUN echo "source /home/ubuntu/simulation_ws/devel/setup.bash" >> /home/ubuntu/.bashrc
RUN echo "export TURTLEBOT3_MODEL=burger" >> /home/ubuntu/.bashrc

# Ensure directories exist for volume mounts
RUN mkdir -p /home/ubuntu/simulation_ws/src/my_turtlebot3_openai_example/trained_models
RUN mkdir -p /home/ubuntu/simulation_ws/src/my_turtlebot3_openai_example/training_reports

ENTRYPOINT ["/bin/bash"]

