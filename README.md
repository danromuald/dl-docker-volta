# dl-docker-volta

Deep learning Docker containers built for Nvidia's Volta GPU on AWS P3 instances.

# 1. Introduction

Nvidia Tesla [V100][1] is the most advanced GPU in the world, and it's now available on-demand on Amazon EC2 [P3 instances][2]. AWS EC2 P3 instances provide a powerful platform for deep learning (DL) and high performance computing (HPC) workloads, with support for up to 8 NVIDIA Tesla V100 GPUs, each containing 5,120 CUDA Cores and 640 Tensor Cores; with 300 GB/s NVLink hyper-mesh interconnect allows GPU-to-GPU communication at high speed and low latency. 

With `dl-docker-volta` , you get a model for building your own deep learning docker-based workbenches on Volta. Docker is ideal if you want to isolate working environments, yet leverage the same dataset across frameworks. It’s early days in deep learning, and there are many frameworks making it easy to build and train neural networks. All of them improving and adding features, constantly. To take advantage of the entire ecosystem and solve your problems faster, you would want an environment that allows you to scale on-demand for the workloads you’re familiar with, while experimenting faster on those you’re not.

# 2. The Approach

At the root, the project has a folder for each deep learning framework. Each framework has a ```Dockerfile```, an ```entrypoint.sh```, and a ```README.md``` file to get you building Docker images fast on the host machine. The ```Dockerfile``` contains all the required code to install and configure a specific deep learning framework on the host machine. The ```entrypoint.sh``` file contains bootstrap actions to be executed every time a container is launched from an image. In this case, checking availability of GPUs. The ```README.md``` file gives you specifics about building the framework. 

After building the images, you'd want to maintain them on a Docker registry. [Amazon ECR][3] allows you to securely store and share your Docker images on a scalable registry.

Here is the folder structure:

```Bash
├── LICENSE
├── README.md
├── mxnet-dlimg
│   └── README.md
├── prepare_ubuntu-xenial-amd64.sh
├── pytorch-dlimg
│   ├── Dockerfile
│   ├── README.md
│   └── entrypoint.sh
└── tensorflow-dlimg
    └── README.md
```

## Preparing the Host

To get started, you need to prepare your host machine, which will run the containers. We use [Ubuntu 16.04 Xenial][4] as the base image. On it, we install the following:

1. A Docker repository
2. Nvidia GPU drivers and plugins
3. ```nvidia-docker```: a wrapper around docker, which checks for GPU availability in the container and automatically fixes character devices corresponding to Nvidia GPUs (```e.g /dev/nvidia0```).

The following code is available in `prepare_ubuntu-xenial-amd64.sh`, and should be executed as a [user data script][5] on an Amazon EC2 P3 instance.


```Bash
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

# Install Nvidia-docker

wget -P /tmp https://github.com/NVIDIA/nvidia-docker/releases/download/v1.0.1/nvidia-docker_1.0.1-1_amd64.deb

dpkg -i /tmp/nvidia-docker_1.0.1-1_amd64.deb

# service nvidia-docker start   # not necessary, reboot at the end.

# Install nvidia-modprobe and nvidia-384 (plugins and drivers)

apt-get install -y nvidia-modprobe nvidia-384

#################### Health check commands ##################
#                                                           #
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

## Todo (FOR YOU): Add the build files (Dockerfile and entrypoint.sh)
##  for each framework you'd like to build, in the ${VOLTA_BUILD_HOME_DIR}

# REBOOT for nvidia plugins and service to take effect

reboot /r

```



[1]:	https://www.nvidia.com/en-us/data-center/tesla-v100/
[2]:	https://aws.amazon.com/about-aws/whats-new/2017/10/introducing-amazon-ec2-p3-instances/
[3]:    https://aws.amazon.com/ecr/
[4]:    https://aws.amazon.com/marketplace/pp/B01JBL2M0O
[5]:    http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/user-data.html
