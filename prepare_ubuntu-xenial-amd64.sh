#!/bin/bash
#
# Notes:
# This script prepares an Ubuntu16.04 (xenial) machine for building
# deep learning docker containers.
# Run as user-data on EC2.
# User data scripts are executed as root user
# so no need to use sudo.

# Vars
__idVer__="17.10:v1.0"
__author__="Dan R. Mbanga"
VOLTA_BUILD_HOME_DIR="/opt/voltaBuild"

# Mbanga, 17.10

set -e

# Add Nvidia repositories for cuda and machine learning
echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list
echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/cuda.list
curl -fsSL http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub |  apt-key add -


# Add Docker repository and install Docker community engine

add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add -

apt-get update

apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    software-properties-common \
    docker-ce


# Other tools

apt-get install -y git \
    tmux \
    htop \
    apt-utils \
    mlocate \
    python3-pip

pip3 install --upgrade pip
pip3 install --upgrade awscli

# Check your keys with ```apt-key --list```

# Nvidia-docker

wget -P /tmp https://github.com/NVIDIA/nvidia-docker/releases/download/v1.0.1/nvidia-docker_1.0.1-1_amd64.deb
dpkg -i /tmp/nvidia-docker_1.0.1-1_amd64.deb
# service nvidia-docker start   # not necessary, reboot at the end.

# Install nvidia-modprobe and nvidia-384 (plugins)

apt-get install -y nvidia-modprobe nvidia-384

#################### Health check commands ##################
#
# journalctl -n -u nvidia-docker                            #
# nvidia-modprobe -u -c=0                                   #
#                                                           #
# Check GPU status on the default nvidia/cuda docker image. #
# nvidia-docker run --rm nvidia/cuda nvidia-smi             #
#                                                           #
#############################################################

# Give user ubuntu access to nvidia-docker and docker binaries
usermod -a -G nvidia-docker ubuntu
usermod -a -G docker ubuntu

############ VOLTA BUILD HOME DIR ######

chmod -R a+w /opt
mkdir -p ${VOLTA_BUILD_HOME_DIR}
chown -R ubuntu:ubuntu ${VOLTA_BUILD_HOME_DIR}

# Update the file locations index
updatedb

## Todo: Add git clone for voltaBuild folder

# git clone the repo.

# REBOOT for nvidia plugins and service to take effect

reboot /r
