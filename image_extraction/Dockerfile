FROM nvidia/cuda:12.0.0-cudnn8-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update -y && apt install -y locales && locale-gen en_US.UTF-8 && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
ENV LANG en_US.UTF-8

RUN apt install -y software-properties-common && add-apt-repository universe && apt update -y && apt install curl -y
RUN apt install nano -y && apt install wget


# PYTHON-PIP
RUN apt update -y && apt install python3-pip -y
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
COPY requirements.txt .
RUN pip3 install -r requirements.txt


# ROS2
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null

RUN apt update -y
RUN DEBIAN_FRONTEND=noninteractive apt install -y ros-dev-tools

RUN apt install ros-humble-desktop -y


# ROS2 WORKSPACE
RUN export COLCON_WS=/home/root/humble_ws && mkdir -p /home/root/humble_ws/src
RUN chmod 777 -R /opt/ros/humble && echo "source /opt/ros/humble/setup.bash" >> /home/root/.bashrc && echo "echo HELLO" >> /home/root/.bashrc

RUN apt update && \
    apt install python3-rosdep -y && \
    rosdep init && \
    rosdep update

RUN /bin/bash -c "source /home/root/.bashrc && \
    cd /home/root/humble_ws/src && \
    ros2 pkg create img_ext --build-type ament_python --license Apache-2.0 && \
    cd /home/root/humble_ws && colcon build"

RUN echo "source /home/root/humble_ws/install/local_setup.bash" >> /home/root/.bashrc

COPY setup.py /home/root/humble_ws/src/img_ext/
COPY package.xml /home/root/humble_ws/src/img_ext/

COPY extract_images.py /home/root/humble_ws/src/img_ext/img_ext/

RUN pip3 install rosbags-image && pip3 install rosbags

RUN cd /home/root/humble_ws && \
    /bin/bash -c "source /home/root/.bashrc" && \
    rosdep install --from-paths src --ignore-src --rosdistro=humble -y && \
    colcon build

RUN echo "source /home/root/humble_ws/install/setup.bash" >> /home/root/.bashrc

