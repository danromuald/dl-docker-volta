# Build Your Own Deep Learning Workspaces on Amazon EC2 P3 instances

Deep learning Docker containers built for Nvidia's Volta GPU on Amazon EC2 P3 instances.



# 1. Introduction

Nvidia Tesla [V100][1] is the most advanced GPU in the world, and it's now available on-demand on Amazon EC2 [P3 instances][2]. AWS EC2 P3 instances provide a powerful platform for deep learning (DL) and high performance computing (HPC) workloads, with support for up to 8 NVIDIA Tesla V100 GPUs, each containing 5,120 CUDA Cores and 640 Tensor Cores; with 300 GB/s NVLink hyper-mesh interconnect allows GPU-to-GPU communication at high speed and low latency. 

With [`dl-docker-volta`][10] , you get a model for building your own deep learning docker-based workbenches on Volta. Docker is ideal if you want to isolate working environments, yet leverage the same dataset across frameworks. It’s early days in deep learning, and there are many frameworks making it easy to build and train neural networks. All of them improving and adding features, constantly. To take advantage of the entire ecosystem and solve your problems faster, you would want an environment that allows you to scale on-demand for the workloads you’re familiar with, while experimenting faster on those you’re not.

# 2. The Approach

At the root, the project has a folder for each deep learning framework. Each framework has a ```Dockerfile```, an ```entrypoint.sh```, and a ```README.md``` file to get you building Docker images fast on the host machine. The ```Dockerfile``` contains all the required code to install and configure a specific deep learning framework on the host machine. The ```entrypoint.sh``` file contains bootstrap actions to be executed every time a container is launched from an image. In this case, checking availability of GPUs. The ```README.md``` file gives you specifics about building the framework. 

After building the images, you'd want to maintain them on a Docker registry. [Amazon ECR][3] allows you to securely store and share your Docker images on a scalable, managed docker images registry.

Here is the folder structure:

```bash
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

# 3. Preparing the host

To get started, you need to prepare your host machine to run the docker containers. I this case, I used [Ubuntu 16.04 Xenial][4] as the base EC2 image. On it, I installed the following:

1. A Docker local repository
2. Nvidia GPU drivers and plugins
3. ```nvidia-docker``` is a wrapper around docker, which checks for GPU availability in the container and automatically fixes character devices corresponding to Nvidia GPUs (```e.g /dev/nvidia0```).
4. AWS CLI
5. Some helper cli tools I find very useful. [tmux][5] is great for maintaining your sessions in a remote machine intact. If you're like me and you always need a reminder of where some files and libraries are,  [mlocate][6] will be very useful. [htop][7] is great for visualizing the resources utilization of your Linux machine.

The following machine preparation code is available in `prepare_ubuntu-xenial-amd64.sh`, and should be executed as a [user data script][8] on an Amazon EC2 P3 instance.


```bash
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

## Todo (HOMEWORK FOR YOU): Add the build files (Dockerfile and entrypoint.sh)
##  for each framework you'd like to build, in the ${VOLTA_BUILD_HOME_DIR}

# REBOOT for nvidia plugins and service to take effect

reboot /r

```

Now you have a working Ubuntu 16.04 machine with Nvidia drivers in it. Next, you have to pull the appropriate folder for the framework you'd want to install on a local container. There is a ```Dockerfile``` in each folder that contains the specifications to ```build``` the image on your host machine. In this case, we'll install PyTorch.

Verify that the Nvidia drivers and plugins are installed, and that the service was started succesfully by ```systemd```.

```bash

ubuntu@ip-10-0-1-216:~$ service nvidia-docker status

● nvidia-docker.service - NVIDIA Docker plugin
   Loaded: loaded (/lib/systemd/system/nvidia-docker.service; enabled; vendor preset: enabled)
   Active: active (running) since Sun 2017-10-29 16:23:21 UTC; 1min 46s ago
     Docs: https://github.com/NVIDIA/nvidia-docker/wiki
  Process: 1338 ExecStartPost=/bin/sh -c /bin/echo unix://$SOCK_DIR/nvidia-docker.sock > $SPEC_FILE (code=exited, status=0/SUCCESS)
  Process: 1327 ExecStartPost=/bin/sh -c /bin/mkdir -p $( dirname $SPEC_FILE ) (code=exited, status=0/SUCCESS)
 Main PID: 1326 (nvidia-docker-p)
    Tasks: 9
   Memory: 28.2M
      CPU: 905ms
   CGroup: /system.slice/nvidia-docker.service
           └─1326 /usr/bin/nvidia-docker-plugin -s /var/lib/nvidia-docker

Oct 29 16:23:21 ip-10-0-1-216 systemd[1]: Starting NVIDIA Docker plugin...
Oct 29 16:23:21 ip-10-0-1-216 systemd[1]: Started NVIDIA Docker plugin.
Oct 29 16:23:21 ip-10-0-1-216 nvidia-docker-plugin[1326]: /usr/bin/nvidia-docker-plugin | 2017/10/29 16:23:21 Loading NVIDIA unified memory
Oct 29 16:23:21 ip-10-0-1-216 nvidia-docker-plugin[1326]: /usr/bin/nvidia-docker-plugin | 2017/10/29 16:23:21 Loading NVIDIA management library
Oct 29 16:23:22 ip-10-0-1-216 nvidia-docker-plugin[1326]: /usr/bin/nvidia-docker-plugin | 2017/10/29 16:23:22 Discovering GPU devices
Oct 29 16:23:22 ip-10-0-1-216 nvidia-docker-plugin[1326]: /usr/bin/nvidia-docker-plugin | 2017/10/29 16:23:22 Provisioning volumes at /var/lib/nvidia-docker/volumes
Oct 29 16:23:22 ip-10-0-1-216 nvidia-docker-plugin[1326]: /usr/bin/nvidia-docker-plugin | 2017/10/29 16:23:22 Serving plugin API at /var/lib/nvidia-docker
Oct 29 16:23:22 ip-10-0-1-216 nvidia-docker-plugin[1326]: /usr/bin/nvidia-docker-plugin | 2017/10/29 16:23:22 Serving remote API at localhost:3476

ubuntu@ip-10-0-1-216:~$ nvidia-docker images -a
nvidia-docker | 2017/10/29 16:25:5
```

Another way to verify it the service is up and running:

```bash

ubuntu@ip-10-0-1-216:~$ journalctl -n -u nvidia-docker

-- Logs begin at Sun 2017-10-29 16:23:17 UTC, end at Sun 2017-10-29 16:24:47 UTC. --
Oct 29 16:23:21 ip-10-0-1-216 systemd[1]: Starting NVIDIA Docker plugin...
Oct 29 16:23:21 ip-10-0-1-216 systemd[1]: Started NVIDIA Docker plugin.
Oct 29 16:23:21 ip-10-0-1-216 nvidia-docker-plugin[1326]: /usr/bin/nvidia-docker-plugin | 2017/10/29 16:23:21 Loading NVIDIA unified memory
Oct 29 16:23:21 ip-10-0-1-216 nvidia-docker-plugin[1326]: /usr/bin/nvidia-docker-plugin | 2017/10/29 16:23:21 Loading NVIDIA management library
Oct 29 16:23:22 ip-10-0-1-216 nvidia-docker-plugin[1326]: /usr/bin/nvidia-docker-plugin | 2017/10/29 16:23:22 Discovering GPU devices
Oct 29 16:23:22 ip-10-0-1-216 nvidia-docker-plugin[1326]: /usr/bin/nvidia-docker-plugin | 2017/10/29 16:23:22 Provisioning volumes at /var/lib/nvidia-docker/volumes
Oct 29 16:23:22 ip-10-0-1-216 nvidia-docker-plugin[1326]: /usr/bin/nvidia-docker-plugin | 2017/10/29 16:23:22 Serving plugin API at /var/lib/nvidia-docker
Oct 29 16:23:22 ip-10-0-1-216 nvidia-docker-plugin[1326]: /usr/bin/nvidia-docker-plugin | 2017/10/29 16:23:22 Serving remote API at localhost:3476
```

Pull the base Nvidia Docker image (```nvidia/cuda:9.0-devel-ubuntu16.04```) which comes with Cuda SDK version 9. This base image will be used by Docker to build the rest of the deep learning machine with PyTorch in it. 

```bash

nvidia-docker run --rm nvidia/cuda nvidia-smi


ubuntu@ip-10-0-1-216:~$ nvidia-docker run --rm nvidia/cuda nvidia-smi
Sun Oct 29 16:28:29 2017       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 384.90                 Driver Version: 384.90                    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla V100-SXM2...  Off  | 00000000:00:1E.0 Off |                    0 |
| N/A   32C    P0    21W / 300W |     10MiB / 16152MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
ubuntu@ip-10-0-1-216:~$ sudo nvidia-docker run --rm 2fa9a0f996e2 nvidia-smi
Sun Oct 29 16:28:42 2017       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 384.90                 Driver Version: 384.90                    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla V100-SXM2...  Off  | 00000000:00:1E.0 Off |                    0 |
| N/A   32C    P0    21W / 300W |     10MiB / 16152MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+

```

In this case I run the container with a ```--rm``` option and a command ```nvidia-smi```; which downloads the image in my local repository, starts a container and runs the ```nvidia-smi``` commands, then exits and removes the container. To look at your images in the local repository, run ```nvidia-docer images -a```. Remember to always use ```nvidia-docker```. It has all the normal ```docker``` functionalities, with the addition of fixing your GPUs for you, nicely!


# 4. Install PyTorch with examples

## 4.1 Prepare the Build environment

Download the ```pytorch-dlimg``` folder under the ```/opt/voltaBuild``` folder in the host EC2 machine. On the host machine, ```cd /opt/voltaBuild/pytorch-dlimg``` and explore teh ```Dockerfile``` and ```entrypoint.sh``` files that will be required to build the ```PyTorch``` image from the base Nvidia CUDA9 image.

Let's review the Dockerfile:


```docker
FROM nvidia/cuda:9.0-devel-ubuntu16.04
LABEL maintainer="Dan R. Mbanga"
# vars
    ## Python version
ENV PYTHON_VERSION 3.5
ENV CUDNN_VERSION 7.0.3.11
LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"
    ## The version of this Docker container
ENV DANULAB_PYTORCH_IMAGE_VERSION 17.10
ENV DANULAB_PYTORCH_BUILD_VERSION 1.0
    ## Dan's Anaconda channel for magma-cuda90. I had to build the package to support cuda9.0.
    ## Not yet available on the default Anaconda repository.
ENV DANULAB_ANACONDA_CHANNEL etcshadow

RUN echo 'export PS1="\[\033[01;32m\]\u@\h \[\033[00m\]: \[\033[01;34m\]\w \[\033[00m\]\$"' >> /etc/profile

ENV PYTORCH_WORK_DIR /pytorch-workspace

###############################
# First, install apt-utils

RUN apt-get update

RUN apt-get install -y apt-utils

RUN echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update

RUN apt-get update && apt-get install -y --no-install-recommends \
            libcudnn7=$CUDNN_VERSION-1+cuda9.0 \
            libcudnn7-dev=$CUDNN_VERSION-1+cuda9.0 \
            build-essential \
            cmake \
            git \
            curl \
            vim \
            tmux \
            mlocate \
            htop \
            ca-certificates \
            libnccl2=2.0.5-3+cuda9.0 \
            libnccl-dev=2.0.5-3+cuda9.0 \
            libjpeg-dev \
            libpng-dev &&\
    rm -rf /var/lib/apt/lists/*

# Install Anaconda full. Not miniconda.
# https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh

RUN curl -o ~/anaconda3-latest.sh -O  https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh  && \
     chmod +x ~/anaconda3-latest.sh && \
     ~/anaconda3-latest.sh -b -p /opt/conda && \
     rm ~/anaconda3-latest.sh && \
     /opt/conda/bin/conda install conda-build && \
     /opt/conda/bin/conda create -y --name pytorch-py35 python=${PYTHON_VERSION} \
        anaconda numpy pyyaml scipy ipython mkl jupyter && \
     /opt/conda/bin/conda clean -ya

ENV PATH /opt/conda/envs/pytorch-py35/bin:$PATH

##### Installing magma-cuda90 from my Anaconda channel ##################################
# https://anaconda.org/etcshadow/magma-cuda90                                           #
#   To build from source (it takes a while!).                                           #
# WORKDIR /workspace                                                                    #
# RUN git clone https://github.com/pytorch/builder.git                                  #
# RUN cp -r builder/conda/magma-cuda90-2.2.0/ magma-cuda90                              #
# WORKDIR /workspace/magma-cuda90                                                       #
# RUN conda-build .                                                                     #
### Go to /opt/conda/conda-bld/magma-cuda90_1507361009645/  (your version might differ) #
# RUN conda install ./magma-cuda90-2.1.0-h865527f_5.tar.bz2                             #
#########################################################################################


RUN conda install --name pytorch-py35 -c ${DANULAB_ANACONDA_CHANNEL} magma-cuda90

WORKDIR /opt

RUN git clone --recursive https://github.com/pytorch/pytorch.git

WORKDIR /opt/pytorch

RUN git submodule update --init

#################################### BUILDING Pytorch for VOLTA on cuda9.0 ##################
# REF: https://github.com/torch/cutorch/blob/master/lib/THC/cmake/select_compute_arch.cmake #
# REF: https://en.wikipedia.org/wiki/CUDA                                                   #
# VOLTA has compute capabilities 7.0 and 7.1                                                #
# CUDA SDK 9.0 supports compute capability 3.0 through 7.x (Kepler, Maxwell, Pascal, Volta) #
#                                                                                           #
# Here we compile for:                                                                      #
#    - Kepler: 3.7 (AWS P2) +PTX, for the P2                                                #
#    - Maxwell: 5.0 5.2                                                                     #
#    - Jetson TX1: 5.3                                                                      #
#    - Pascal P100: 6.0                                                                     #
#    - Pascal GTX family: 6.1                                                               #
#    - Jetson TX2: 6.2                                                                      #
#    - Volta V100: 7.0+PTX (PTX = parallel Thread Execution): even faster!                  #
#############################################################################################

RUN TORCH_CUDA_ARCH_LIST="3.7+PTX 5.0 5.2 5.3 6.0 6.1+PTX 6.2 7.0+PTX" TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    pip install -v .

# Setup the entrypoint.
WORKDIR /opt/pytorch-init

COPY ./entrypoint.sh /opt/pytorch-init/

RUN chmod +x /opt/pytorch-init/entrypoint.sh

# Working directory + examples

WORKDIR ${PYTORCH_WORK_DIR}

RUN git clone --recursive https://github.com/pytorch/examples.git

RUN git clone --recursive https://github.com/pytorch/vision.git && \
    cd vision && pip install -v .

RUN git clone --recursive https://github.com/pytorch/text.git && \
    cd text && pip install -v .

RUN git clone --recursive https://github.com/pytorch/tutorials.git

RUN chmod -R a+w ${PYTORCH_WORK_DIR}

# A reference for exposing jupyter notebooks
EXPOSE 8888

# This helps you easily locate files on the box with ```mlocate <filename>```

RUN updatedb

ENTRYPOINT ["/bin/bash"]

CMD ["/opt/pytorch-init/entrypoint.sh"]
```

A few things happen here:

1. We use the ```nvidia-cuda:9.0-devel-ubuntu16.04``` base image to build our docker environment. This machine comes with CUDA SDK 9.0. The CUDA SDK 9.0 requires CUDNN version 7+ to operate properly. So we install ```libcudnn7``` and ```libcudnn7-dev``` with the appropriate ```${CUDNN_VERSION}```. 


2. CUDA 9.0 also requires ```magma-cuda90``` to accelerate linear algebra operations on Volta. magma-cuda90 isn't yet available on Anaconda default repositories, so we built a version for Volta which is available on my channel [```etcshadow```][9]. The steps to build ```magma-cuda90``` from source are available as comments in the ```Dockerfile```.


3. We install the full version of ```anaconda3``` (you could also consider ```miniconda```, I like having all packages available by default); then create a conda environment for ```PyTorch``` called ```pytorch-python35```, with full ```anaconda3``` loaded, ```python3.5```, extra libraries for data science, and ```mkl``` for math kernel acceleration on CPU.

4. We download and build ```pytorch``` for the following Nvidia GPUs:

- Kepler (AWS P2 instances)
- Maxwell
- Jetson TX1 and TX2
- Pascal P100
- Volta V100

5. Finally, we create a working directory under ```/pytorch-workspace``` where we put pytorch examples.


## 4.2 Build the Docker image

Move to the ```/opt/voltaBuild/pytorch-dlimg/``` folder, and run ```nvidia-docker build -t <image-name>:<tag-name> .``` to build the image. 

In this case, I ran: 

```bash

nvidia-docker build -t danulab/pytorch-dlimg:17.10 -t danulab/pytorch-
dlimg:v1.0 -t danulab/pytorch-dlimg:latest .

```

You may want to name your images differently, in my case, I named it ```pytorch-dlimg``` with tags ```17.10``` and ```latest```. You may also want to go get dinner or breakfast because the image build takes about 1.5 hours to complete on a ```p2.2xlarge```.
Once the build is completed succesfully, you will get confirmation messages. I've pasted here the last 3 lines of my build.

```bash
Successfully tagged danulab/pytorch-dlimg:17.10
Successfully tagged danulab/pytorch-dlimg:v1.0
Successfully tagged danulab/pytorch-dlimg:latest
```

## 4.3 Run the Docker Container

Congrats! You now have a running docker environment, built for Volta, with Cuda9 and a pytorch image loaded with Anaconda3, and lots of examples. To view your images, run ```nvidia-docker images -a```. To run an interactive session of your new docker container, do ```nvidia-docker run -it <image-name>```. In my case:

```bash

# Run the docker container in interactive mode.

ubuntu@ip-10-0-1-216:~$ nvidia-docker run -it  danulab-pytorch 

root@a780acd1a36d:/python-workspace# 

# Activate pytorch-python35 environment
root@a780acd1a36d:/pytorch-workspace# source activate /opt/conda/envs/pytorch-py35/
(/opt/conda/envs/pytorch-py35/) root@a780acd1a36d:/pytorch-workspace# 

# Moving to the examples folder
root@a780acd1a36d:/pytorch-workspace/examples/mnist# cd examples/mnist/
root@a780acd1a36d:/pytorch-workspace/examples/mnist#

```

# 5. Deep Learning at last!

Okay, we did all of this to enjoy the speed of Nvidia Volta on Amazon EC2 P3 instances. It's time to test it!. Move to the examples folder under your workspace in ```/pytorch-workspace/exampels/```, and pick the one you want to execute. I suggest you time it to see how fast it is. Here is how I ran the traditional ```mnist``` example in ```1 min 15 seconds```, or an average processing speed of ```60K images every 7.5 seconds!```. The long and boring log below is for the ```10 Epochs``` or iterations on the entire ```mnist``` dataset. I keep it here so you get a feel of the reality of training deep neural networks :-). For your projects, you will likely go ```50 Epochs``` or more. That's why you'd want a tool like ```tmux``` to start a session, launch the job, then disconnect and reconnect into the same session, at will! 


```bash

# timing the mnist example -- second run. The first run downloads the data.

(/opt/conda/envs/pytorch-py35/) root@a780acd1a36d:/pytorch-workspace/examples/mnist# time python main.py 

main.py:68: UserWarning: Implicit dimension choice for log_softmax has been deprecated. Change the call to include dim=X as an argument.
  return F.log_softmax(x)
Train Epoch: 1 [0/60000 (0%)]	Loss: 2.390087
Train Epoch: 1 [640/60000 (1%)]	Loss: 2.350225
Train Epoch: 1 [1280/60000 (2%)]	Loss: 2.288934
Train Epoch: 1 [1920/60000 (3%)]	Loss: 2.279396
Train Epoch: 1 [2560/60000 (4%)]	Loss: 2.272692
Train Epoch: 1 [3200/60000 (5%)]	Loss: 2.279640
Train Epoch: 1 [3840/60000 (6%)]	Loss: 2.276944
Train Epoch: 1 [4480/60000 (7%)]	Loss: 2.233707
Train Epoch: 1 [5120/60000 (9%)]	Loss: 2.222649
Train Epoch: 1 [5760/60000 (10%)]	Loss: 2.181699
Train Epoch: 1 [6400/60000 (11%)]	Loss: 2.159016
Train Epoch: 1 [7040/60000 (12%)]	Loss: 2.055780
Train Epoch: 1 [7680/60000 (13%)]	Loss: 2.065891
Train Epoch: 1 [8320/60000 (14%)]	Loss: 1.916489
Train Epoch: 1 [8960/60000 (15%)]	Loss: 1.909557
Train Epoch: 1 [9600/60000 (16%)]	Loss: 1.704020
Train Epoch: 1 [10240/60000 (17%)]	Loss: 1.555293
Train Epoch: 1 [10880/60000 (18%)]	Loss: 1.621865
Train Epoch: 1 [11520/60000 (19%)]	Loss: 1.522296
Train Epoch: 1 [12160/60000 (20%)]	Loss: 1.533320
Train Epoch: 1 [12800/60000 (21%)]	Loss: 1.338756
Train Epoch: 1 [13440/60000 (22%)]	Loss: 1.387478
Train Epoch: 1 [14080/60000 (23%)]	Loss: 1.332887
Train Epoch: 1 [14720/60000 (25%)]	Loss: 1.268512
Train Epoch: 1 [15360/60000 (26%)]	Loss: 1.082959
Train Epoch: 1 [16000/60000 (27%)]	Loss: 1.206841
Train Epoch: 1 [16640/60000 (28%)]	Loss: 0.873271
Train Epoch: 1 [17280/60000 (29%)]	Loss: 0.775382
Train Epoch: 1 [17920/60000 (30%)]	Loss: 0.975311
Train Epoch: 1 [18560/60000 (31%)]	Loss: 0.907628
Train Epoch: 1 [19200/60000 (32%)]	Loss: 1.190366
Train Epoch: 1 [19840/60000 (33%)]	Loss: 0.861021
Train Epoch: 1 [20480/60000 (34%)]	Loss: 0.889621
Train Epoch: 1 [21120/60000 (35%)]	Loss: 1.076115
Train Epoch: 1 [21760/60000 (36%)]	Loss: 0.968419
Train Epoch: 1 [22400/60000 (37%)]	Loss: 0.722111
Train Epoch: 1 [23040/60000 (38%)]	Loss: 0.864212
Train Epoch: 1 [23680/60000 (39%)]	Loss: 0.828237
Train Epoch: 1 [24320/60000 (41%)]	Loss: 0.798611
Train Epoch: 1 [24960/60000 (42%)]	Loss: 0.738692
Train Epoch: 1 [25600/60000 (43%)]	Loss: 0.582205
Train Epoch: 1 [26240/60000 (44%)]	Loss: 0.725970
Train Epoch: 1 [26880/60000 (45%)]	Loss: 0.755889
Train Epoch: 1 [27520/60000 (46%)]	Loss: 0.513736
Train Epoch: 1 [28160/60000 (47%)]	Loss: 0.767814
Train Epoch: 1 [28800/60000 (48%)]	Loss: 0.604210
Train Epoch: 1 [29440/60000 (49%)]	Loss: 0.568367
Train Epoch: 1 [30080/60000 (50%)]	Loss: 0.748543
Train Epoch: 1 [30720/60000 (51%)]	Loss: 0.798787
Train Epoch: 1 [31360/60000 (52%)]	Loss: 0.825547
Train Epoch: 1 [32000/60000 (53%)]	Loss: 0.650468
Train Epoch: 1 [32640/60000 (54%)]	Loss: 0.867096
Train Epoch: 1 [33280/60000 (55%)]	Loss: 0.443044
Train Epoch: 1 [33920/60000 (57%)]	Loss: 0.663155
Train Epoch: 1 [34560/60000 (58%)]	Loss: 0.588300
Train Epoch: 1 [35200/60000 (59%)]	Loss: 0.540387
Train Epoch: 1 [35840/60000 (60%)]	Loss: 0.496390
Train Epoch: 1 [36480/60000 (61%)]	Loss: 0.495418
Train Epoch: 1 [37120/60000 (62%)]	Loss: 0.522647
Train Epoch: 1 [37760/60000 (63%)]	Loss: 0.672931
Train Epoch: 1 [38400/60000 (64%)]	Loss: 0.759644
Train Epoch: 1 [39040/60000 (65%)]	Loss: 0.781511
Train Epoch: 1 [39680/60000 (66%)]	Loss: 0.791905
Train Epoch: 1 [40320/60000 (67%)]	Loss: 0.713253
Train Epoch: 1 [40960/60000 (68%)]	Loss: 0.601441
Train Epoch: 1 [41600/60000 (69%)]	Loss: 0.839365
Train Epoch: 1 [42240/60000 (70%)]	Loss: 0.467135
Train Epoch: 1 [42880/60000 (71%)]	Loss: 0.660879
Train Epoch: 1 [43520/60000 (72%)]	Loss: 0.619761
Train Epoch: 1 [44160/60000 (74%)]	Loss: 0.813196
Train Epoch: 1 [44800/60000 (75%)]	Loss: 0.483893
Train Epoch: 1 [45440/60000 (76%)]	Loss: 0.524351
Train Epoch: 1 [46080/60000 (77%)]	Loss: 0.543117
Train Epoch: 1 [46720/60000 (78%)]	Loss: 0.310266
Train Epoch: 1 [47360/60000 (79%)]	Loss: 0.491195
Train Epoch: 1 [48000/60000 (80%)]	Loss: 0.638367
Train Epoch: 1 [48640/60000 (81%)]	Loss: 0.437589
Train Epoch: 1 [49280/60000 (82%)]	Loss: 0.341376
Train Epoch: 1 [49920/60000 (83%)]	Loss: 0.521301
Train Epoch: 1 [50560/60000 (84%)]	Loss: 0.543150
Train Epoch: 1 [51200/60000 (85%)]	Loss: 0.370414
Train Epoch: 1 [51840/60000 (86%)]	Loss: 0.591696
Train Epoch: 1 [52480/60000 (87%)]	Loss: 0.596892
Train Epoch: 1 [53120/60000 (88%)]	Loss: 0.444391
Train Epoch: 1 [53760/60000 (90%)]	Loss: 0.522760
Train Epoch: 1 [54400/60000 (91%)]	Loss: 0.577707
Train Epoch: 1 [55040/60000 (92%)]	Loss: 0.374454
Train Epoch: 1 [55680/60000 (93%)]	Loss: 0.694439
Train Epoch: 1 [56320/60000 (94%)]	Loss: 0.526622
Train Epoch: 1 [56960/60000 (95%)]	Loss: 0.536326
Train Epoch: 1 [57600/60000 (96%)]	Loss: 0.401314
Train Epoch: 1 [58240/60000 (97%)]	Loss: 0.374374
Train Epoch: 1 [58880/60000 (98%)]	Loss: 0.312261
Train Epoch: 1 [59520/60000 (99%)]	Loss: 0.574444

Test set: Average loss: 0.2046, Accuracy: 9408/10000 (94%)

Train Epoch: 2 [0/60000 (0%)]	Loss: 0.486073
Train Epoch: 2 [640/60000 (1%)]	Loss: 0.530259
Train Epoch: 2 [1280/60000 (2%)]	Loss: 0.862629
Train Epoch: 2 [1920/60000 (3%)]	Loss: 0.493156
Train Epoch: 2 [2560/60000 (4%)]	Loss: 0.328225
Train Epoch: 2 [3200/60000 (5%)]	Loss: 0.275720
Train Epoch: 2 [3840/60000 (6%)]	Loss: 0.240274
Train Epoch: 2 [4480/60000 (7%)]	Loss: 0.387968
Train Epoch: 2 [5120/60000 (9%)]	Loss: 0.304994
Train Epoch: 2 [5760/60000 (10%)]	Loss: 0.312698
Train Epoch: 2 [6400/60000 (11%)]	Loss: 0.337541
Train Epoch: 2 [7040/60000 (12%)]	Loss: 0.368442
Train Epoch: 2 [7680/60000 (13%)]	Loss: 0.414195
Train Epoch: 2 [8320/60000 (14%)]	Loss: 0.390866
Train Epoch: 2 [8960/60000 (15%)]	Loss: 0.589836
Train Epoch: 2 [9600/60000 (16%)]	Loss: 0.377261
Train Epoch: 2 [10240/60000 (17%)]	Loss: 0.536778
Train Epoch: 2 [10880/60000 (18%)]	Loss: 0.547609
Train Epoch: 2 [11520/60000 (19%)]	Loss: 0.302692
Train Epoch: 2 [12160/60000 (20%)]	Loss: 0.371663
Train Epoch: 2 [12800/60000 (21%)]	Loss: 0.331667
Train Epoch: 2 [13440/60000 (22%)]	Loss: 0.324786
Train Epoch: 2 [14080/60000 (23%)]	Loss: 0.520558
Train Epoch: 2 [14720/60000 (25%)]	Loss: 0.355960
Train Epoch: 2 [15360/60000 (26%)]	Loss: 0.618069
Train Epoch: 2 [16000/60000 (27%)]	Loss: 0.658417
Train Epoch: 2 [16640/60000 (28%)]	Loss: 0.295800
Train Epoch: 2 [17280/60000 (29%)]	Loss: 0.198287
Train Epoch: 2 [17920/60000 (30%)]	Loss: 0.498658
Train Epoch: 2 [18560/60000 (31%)]	Loss: 0.444633
Train Epoch: 2 [19200/60000 (32%)]	Loss: 0.406007
Train Epoch: 2 [19840/60000 (33%)]	Loss: 0.316711
Train Epoch: 2 [20480/60000 (34%)]	Loss: 0.388651
Train Epoch: 2 [21120/60000 (35%)]	Loss: 0.205107
Train Epoch: 2 [21760/60000 (36%)]		Loss: 0.444495
Train Epoch: 2 [22400/60000 (37%)]	Loss: 0.421520
Train Epoch: 2 [23040/60000 (38%)]	Loss: 0.295586
Train Epoch: 2 [23680/60000 (39%)]	Loss: 0.345527
Train Epoch: 2 [24320/60000 (41%)]	Loss: 0.356803
Train Epoch: 2 [24960/60000 (42%)]	Loss: 0.523780
Train Epoch: 2 [25600/60000 (43%)]	Loss: 0.487095
Train Epoch: 2 [26240/60000 (44%)]	Loss: 0.575701
Train Epoch: 2 [26880/60000 (45%)]	Loss: 0.376640
Train Epoch: 2 [27520/60000 (46%)]	Loss: 0.336968
Train Epoch: 2 [28160/60000 (47%)]	Loss: 0.403694
Train Epoch: 2 [28800/60000 (48%)]	Loss: 0.219041
Train Epoch: 2 [29440/60000 (49%)]	Loss: 0.595567
Train Epoch: 2 [30080/60000 (50%)]	Loss: 0.482775
Train Epoch: 2 [30720/60000 (51%)]	Loss: 0.281583
Train Epoch: 2 [31360/60000 (52%)]	Loss: 0.354142
Train Epoch: 2 [32000/60000 (53%)]	Loss: 0.384513
Train Epoch: 2 [32640/60000 (54%)]	Loss: 0.329443
Train Epoch: 2 [33280/60000 (55%)]	Loss: 0.331024
Train Epoch: 2 [33920/60000 (57%)]	Loss: 0.408204
Train Epoch: 2 [34560/60000 (58%)]	Loss: 0.507622
Train Epoch: 2 [35200/60000 (59%)]	Loss: 0.321690
Train Epoch: 2 [35840/60000 (60%)]	Loss: 0.295482
Train Epoch: 2 [36480/60000 (61%)]	Loss: 0.567883
Train Epoch: 2 [37120/60000 (62%)]	Loss: 0.339266
Train Epoch: 2 [37760/60000 (63%)]	Loss: 0.328928
Train Epoch: 2 [38400/60000 (64%)]	Loss: 0.366879
Train Epoch: 2 [39040/60000 (65%)]	Loss: 0.399510
Train Epoch: 2 [39680/60000 (66%)]	Loss: 0.348691
Train Epoch: 2 [40320/60000 (67%)]	Loss: 0.227425
Train Epoch: 2 [40960/60000 (68%)]	Loss: 0.318275
Train Epoch: 2 [41600/60000 (69%)]	Loss: 0.500673
Train Epoch: 2 [42240/60000 (70%)]	Loss: 0.263500
Train Epoch: 2 [42880/60000 (71%)]	Loss: 0.309098
Train Epoch: 2 [43520/60000 (72%)]	Loss: 0.353623
Train Epoch: 2 [44160/60000 (74%)]	Loss: 0.476302
Train Epoch: 2 [44800/60000 (75%)]	Loss: 0.473411
Train Epoch: 2 [45440/60000 (76%)]	Loss: 0.384841
Train Epoch: 2 [46080/60000 (77%)]	Loss: 0.447103
Train Epoch: 2 [46720/60000 (78%)]	Loss: 0.282055
Train Epoch: 2 [47360/60000 (79%)]	Loss: 0.175965
Train Epoch: 2 [48000/60000 (80%)]	Loss: 0.360219
Train Epoch: 2 [48640/60000 (81%)]	Loss: 0.330120
Train Epoch: 2 [49280/60000 (82%)]	Loss: 0.416644
Train Epoch: 2 [49920/60000 (83%)]	Loss: 0.432739
Train Epoch: 2 [50560/60000 (84%)]	Loss: 0.377681
Train Epoch: 2 [51200/60000 (85%)]	Loss: 0.312238
Train Epoch: 2 [51840/60000 (86%)]	Loss: 0.326372
Train Epoch: 2 [52480/60000 (87%)]	Loss: 0.251546
Train Epoch: 2 [53120/60000 (88%)]	Loss: 0.392894
Train Epoch: 2 [53760/60000 (90%)]	Loss: 0.229773
Train Epoch: 2 [54400/60000 (91%)]	Loss: 0.327291
Train Epoch: 2 [55040/60000 (92%)]	Loss: 0.596074
Train Epoch: 2 [55680/60000 (93%)]	Loss: 0.364597
Train Epoch: 2 [56320/60000 (94%)]	Loss: 0.567585
Train Epoch: 2 [56960/60000 (95%)]	Loss: 0.515399
Train Epoch: 2 [57600/60000 (96%)]	Loss: 0.313139
Train Epoch: 2 [58240/60000 (97%)]	Loss: 0.273254
Train Epoch: 2 [58880/60000 (98%)]	Loss: 0.291174
Train Epoch: 2 [59520/60000 (99%)]	Loss: 0.562903

Test set: Average loss: 0.1272, Accuracy: 9611/10000 (96%)

Train Epoch: 3 [0/60000 (0%)]	Loss: 0.248993
Train Epoch: 3 [640/60000 (1%)]	Loss: 0.376218
Train Epoch: 3 [1280/60000 (2%)]	Loss: 0.187322
Train Epoch: 3 [1920/60000 (3%)]	Loss: 0.233307
Train Epoch: 3 [2560/60000 (4%)]	Loss: 0.428122
Train Epoch: 3 [3200/60000 (5%)]	Loss: 0.293849
Train Epoch: 3 [3840/60000 (6%)]	Loss: 0.358833
Train Epoch: 3 [4480/60000 (7%)]	Loss: 0.340423
Train Epoch: 3 [5120/60000 (9%)]	Loss: 0.239373
Train Epoch: 3 [5760/60000 (10%)]	Loss: 0.347159
Train Epoch: 3 [6400/60000 (11%)]	Loss: 0.366581
Train Epoch: 3 [7040/60000 (12%)]	Loss: 0.230147
Train Epoch: 3 [7680/60000 (13%)]	Loss: 0.399727
Train Epoch: 3 [8320/60000 (14%)]	Loss: 0.248005
Train Epoch: 3 [8960/60000 (15%)]	Loss: 0.322460
Train Epoch: 3 [9600/60000 (16%)]	Loss: 0.461747
Train Epoch: 3 [10240/60000 (17%)]	Loss: 0.311829
Train Epoch: 3 [10880/60000 (18%)]	Loss: 0.371648
Train Epoch: 3 [11520/60000 (19%)]	Loss: 0.364546
Train Epoch: 3 [12160/60000 (20%)]	Loss: 0.196105
Train Epoch: 3 [12800/60000 (21%)]	Loss: 0.291166
Train Epoch: 3 [13440/60000 (22%)]	Loss: 0.136354
Train Epoch: 3 [14080/60000 (23%)]	Loss: 0.365807
Train Epoch: 3 [14720/60000 (25%)]	Loss: 0.397847
Train Epoch: 3 [15360/60000 (26%)]	Loss: 0.240663
Train Epoch: 3 [16000/60000 (27%)]	Loss: 0.288483
Train Epoch: 3 [16640/60000 (28%)]	Loss: 0.278308
Train Epoch: 3 [17280/60000 (29%)]	Loss: 0.186496
Train Epoch: 3 [17920/60000 (30%)]	Loss: 0.231048
Train Epoch: 3 [18560/60000 (31%)]	Loss: 0.450305
Train Epoch: 3 [19200/60000 (32%)]	Loss: 0.262193
Train Epoch: 3 [19840/60000 (33%)]	Loss: 0.410101
Train Epoch: 3 [20480/60000 (34%)]	Loss: 0.355595
Train Epoch: 3 [21120/60000 (35%)]	Loss: 0.481436
Train Epoch: 3 [21760/60000 (36%)]	Loss: 0.385373
Train Epoch: 3 [22400/60000 (37%)]	Loss: 0.316759
Train Epoch: 3 [23040/60000 (38%)]	Loss: 0.375975
Train Epoch: 3 [23680/60000 (39%)]	Loss: 0.339003
Train Epoch: 3 [24320/60000 (41%)]	Loss: 0.400077
Train Epoch: 3 [24960/60000 (42%)]	Loss: 0.382375
Train Epoch: 3 [25600/60000 (43%)]	Loss: 0.159524
Train Epoch: 3 [26240/60000 (44%)]	Loss: 0.245008
Train Epoch: 3 [26880/60000 (45%)]	Loss: 0.298296
Train Epoch: 3 [27520/60000 (46%)]	Loss: 0.255043
Train Epoch: 3 [28160/60000 (47%)]	Loss: 0.281519
Train Epoch: 3 [28800/60000 (48%)]	Loss: 0.258827
Train Epoch: 3 [29440/60000 (49%)]	Loss: 0.274323
Train Epoch: 3 [30080/60000 (50%)]	Loss: 0.289882
Train Epoch: 3 [30720/60000 (51%)]	Loss: 0.539577
Train Epoch: 3 [31360/60000 (52%)]	Loss: 0.318197
Train Epoch: 3 [32000/60000 (53%)]	Loss: 0.361184
Train Epoch: 3 [32640/60000 (54%)]	Loss: 0.295282
Train Epoch: 3 [33280/60000 (55%)]	Loss: 0.283397
Train Epoch: 3 [33920/60000 (57%)]	Loss: 0.197356
Train Epoch: 3 [34560/60000 (58%)]	Loss: 0.392582
Train Epoch: 3 [35200/60000 (59%)]	Loss: 0.129394
Train Epoch: 3 [35840/60000 (60%)]	Loss: 0.270159
Train Epoch: 3 [36480/60000 (61%)]	Loss: 0.384386
Train Epoch: 3 [37120/60000 (62%)]	Loss: 0.224063
Train Epoch: 3 [37760/60000 (63%)]	Loss: 0.382079
Train Epoch: 3 [38400/60000 (64%)]	Loss: 0.159203
Train Epoch: 3 [39040/60000 (65%)]	Loss: 0.239889
Train Epoch: 3 [39680/60000 (66%)]	Loss: 0.265286
Train Epoch: 3 [40320/60000 (67%)]	Loss: 0.293538
Train Epoch: 3 [40960/60000 (68%)]	Loss: 0.354138
Train Epoch: 3 [41600/60000 (69%)]	Loss: 0.292805
Train Epoch: 3 [42240/60000 (70%)]	Loss: 0.300338
Train Epoch: 3 [42880/60000 (71%)]	Loss: 0.246825
Train Epoch: 3 [43520/60000 (72%)]	Loss: 0.518568
Train Epoch: 3 [44160/60000 (74%)]	Loss: 0.407816
Train Epoch: 3 [44800/60000 (75%)]	Loss: 0.543875
Train Epoch: 3 [45440/60000 (76%)]	Loss: 0.302905
Train Epoch: 3 [46080/60000 (77%)]	Loss: 0.424616
Train Epoch: 3 [46720/60000 (78%)]	Loss: 0.344024
Train Epoch: 3 [47360/60000 (79%)]	Loss: 0.222284
Train Epoch: 3 [48000/60000 (80%)]	Loss: 0.223708
Train Epoch: 3 [48640/60000 (81%)]	Loss: 0.274509
Train Epoch: 3 [49280/60000 (82%)]	Loss: 0.325061
Train Epoch: 3 [49920/60000 (83%)]	Loss: 0.308782
Train Epoch: 3 [50560/60000 (84%)]	Loss: 0.163518
Train Epoch: 3 [51200/60000 (85%)]	Loss: 0.289852
Train Epoch: 3 [51840/60000 (86%)]	Loss: 0.177998
Train Epoch: 3 [52480/60000 (87%)]	Loss: 0.241937
Train Epoch: 3 [53120/60000 (88%)]	Loss: 0.422326
Train Epoch: 3 [53760/60000 (90%)]	Loss: 0.229395
Train Epoch: 3 [54400/60000 (91%)]	Loss: 0.165190
Train Epoch: 3 [55040/60000 (92%)]	Loss: 0.454961
Train Epoch: 3 [55680/60000 (93%)]	Loss: 0.441349
Train Epoch: 3 [56320/60000 (94%)]	Loss: 0.281250
Train Epoch: 3 [56960/60000 (95%)]	Loss: 0.251524
Train Epoch: 3 [57600/60000 (96%)]	Loss: 0.327538
Train Epoch: 3 [58240/60000 (97%)]	Loss: 0.126168
Train Epoch: 3 [58880/60000 (98%)]	Loss: 0.418447
Train Epoch: 3 [59520/60000 (99%)]	Loss: 0.200302

Test set: Average loss: 0.0993, Accuracy: 9687/10000 (97%)

Train Epoch: 4 [0/60000 (0%)]	Loss: 0.215602
Train Epoch: 4 [640/60000 (1%)]	Loss: 0.219196
Train Epoch: 4 [1280/60000 (2%)]	Loss: 0.243329
Train Epoch: 4 [1920/60000 (3%)]	Loss: 0.339690
Train Epoch: 4 [2560/60000 (4%)]	Loss: 0.295232
Train Epoch: 4 [3200/60000 (5%)]	Loss: 0.307780
Train Epoch: 4 [3840/60000 (6%)]	Loss: 0.418646
Train Epoch: 4 [4480/60000 (7%)]	Loss: 0.215234
Train Epoch: 4 [5120/60000 (9%)]	Loss: 0.350101
Train Epoch: 4 [5760/60000 (10%)]	Loss: 0.203215
Train Epoch: 4 [6400/60000 (11%)]	Loss: 0.145945
Train Epoch: 4 [7040/60000 (12%)]	Loss: 0.416578
Train Epoch: 4 [7680/60000 (13%)]	Loss: 0.261097
Train Epoch: 4 [8320/60000 (14%)]	Loss: 0.352918
Train Epoch: 4 [8960/60000 (15%)]	Loss: 0.322533
Train Epoch: 4 [9600/60000 (16%)]	Loss: 0.213272
Train Epoch: 4 [10240/60000 (17%)]	Loss: 0.293111
Train Epoch: 4 [10880/60000 (18%)]	Loss: 0.290085
Train Epoch: 4 [11520/60000 (19%)]	Loss: 0.206643
Train Epoch: 4 [12160/60000 (20%)]	Loss: 0.371910
Train Epoch: 4 [12800/60000 (21%)]	Loss: 0.182627
Train Epoch: 4 [13440/60000 (22%)]	Loss: 0.228208
Train Epoch: 4 [14080/60000 (23%)]	Loss: 0.465501
Train Epoch: 4 [14720/60000 (25%)]	Loss: 0.531131
Train Epoch: 4 [15360/60000 (26%)]	Loss: 0.222574
Train Epoch: 4 [16000/60000 (27%)]	Loss: 0.129866
Train Epoch: 4 [16640/60000 (28%)]	Loss: 0.400227
Train Epoch: 4 [17280/60000 (29%)]	Loss: 0.458943
Train Epoch: 4 [17920/60000 (30%)]	Loss: 0.285130
Train Epoch: 4 [18560/60000 (31%)]	Loss: 0.219288
Train Epoch: 4 [19200/60000 (32%)]	Loss: 0.140704
Train Epoch: 4 [19840/60000 (33%)]	Loss: 0.186290
Train Epoch: 4 [20480/60000 (34%)]	Loss: 0.111028
Train Epoch: 4 [21120/60000 (35%)]	Loss: 0.286887
Train Epoch: 4 [21760/60000 (36%)]	Loss: 0.416436
Train Epoch: 4 [22400/60000 (37%)]	Loss: 0.298940
Train Epoch: 4 [23040/60000 (38%)]	Loss: 0.313525
Train Epoch: 4 [23680/60000 (39%)]	Loss: 0.270084
Train Epoch: 4 [24320/60000 (41%)]	Loss: 0.308344
Train Epoch: 4 [24960/60000 (42%)]	Loss: 0.394905
Train Epoch: 4 [25600/60000 (43%)]	Loss: 0.200596
Train Epoch: 4 [26240/60000 (44%)]	Loss: 0.375551
Train Epoch: 4 [26880/60000 (45%)]	Loss: 0.311545
Train Epoch: 4 [27520/60000 (46%)]	Loss: 0.297756
Train Epoch: 4 [28160/60000 (47%)]	Loss: 0.382918
Train Epoch: 4 [28800/60000 (48%)]	Loss: 0.285870
Train Epoch: 4 [29440/60000 (49%)]	Loss: 0.259264
Train Epoch: 4 [30080/60000 (50%)]	Loss: 0.205102
Train Epoch: 4 [30720/60000 (51%)]	Loss: 0.366557
Train Epoch: 4 [31360/60000 (52%)]	Loss: 0.295933
Train Epoch: 4 [32000/60000 (53%)]	Loss: 0.209784
Train Epoch: 4 [32640/60000 (54%)]	Loss: 0.421265
Train Epoch: 4 [33280/60000 (55%)]	Loss: 0.262944
Train Epoch: 4 [33920/60000 (57%)]	Loss: 0.128379
Train Epoch: 4 [34560/60000 (58%)]	Loss: 0.211679
Train Epoch: 4 [35200/60000 (59%)]	Loss: 0.303003
Train Epoch: 4 [35840/60000 (60%)]	Loss: 0.158420
Train Epoch: 4 [36480/60000 (61%)]	Loss: 0.324791
Train Epoch: 4 [37120/60000 (62%)]	Loss: 0.361513
Train Epoch: 4 [37760/60000 (63%)]	Loss: 0.256148
Train Epoch: 4 [38400/60000 (64%)]	Loss: 0.152686
Train Epoch: 4 [39040/60000 (65%)]	Loss: 0.225571
Train Epoch: 4 [39680/60000 (66%)]	Loss: 0.445994
Train Epoch: 4 [40320/60000 (67%)]	Loss: 0.098441
Train Epoch: 4 [40960/60000 (68%)]	Loss: 0.083215
Train Epoch: 4 [41600/60000 (69%)]	Loss: 0.233999
Train Epoch: 4 [42240/60000 (70%)]	Loss: 0.148909
Train Epoch: 4 [42880/60000 (71%)]	Loss: 0.225322
Train Epoch: 4 [43520/60000 (72%)]	Loss: 0.232643
Train Epoch: 4 [44160/60000 (74%)]	Loss: 0.307072
Train Epoch: 4 [44800/60000 (75%)]	Loss: 0.619514
Train Epoch: 4 [45440/60000 (76%)]	Loss: 0.329400
Train Epoch: 4 [46080/60000 (77%)]	Loss: 0.077596
Train Epoch: 4 [46720/60000 (78%)]	Loss: 0.200468
Train Epoch: 4 [47360/60000 (79%)]	Loss: 0.373011
Train Epoch: 4 [48000/60000 (80%)]	Loss: 0.247282
Train Epoch: 4 [48640/60000 (81%)]	Loss: 0.146582
Train Epoch: 4 [49280/60000 (82%)]	Loss: 0.489037
Train Epoch: 4 [49920/60000 (83%)]	Loss: 0.198177
Train Epoch: 4 [50560/60000 (84%)]	Loss: 0.211014
Train Epoch: 4 [51200/60000 (85%)]	Loss: 0.090135
Train Epoch: 4 [51840/60000 (86%)]	Loss: 0.152396
Train Epoch: 4 [52480/60000 (87%)]	Loss: 0.422666
Train Epoch: 4 [53120/60000 (88%)]	Loss: 0.170053
Train Epoch: 4 [53760/60000 (90%)]	Loss: 0.160002
Train Epoch: 4 [54400/60000 (91%)]	Loss: 0.102438
Train Epoch: 4 [55040/60000 (92%)]	Loss: 0.204945
Train Epoch: 4 [55680/60000 (93%)]	Loss: 0.331762
Train Epoch: 4 [56320/60000 (94%)]	Loss: 0.283450
Train Epoch: 4 [56960/60000 (95%)]	Loss: 0.209699
Train Epoch: 4 [57600/60000 (96%)]	Loss: 0.397218
Train Epoch: 4 [58240/60000 (97%)]	Loss: 0.438639
Train Epoch: 4 [58880/60000 (98%)]	Loss: 0.125181
Train Epoch: 4 [59520/60000 (99%)]	Loss: 0.223498

Test set: Average loss: 0.0867, Accuracy: 9721/10000 (97%)

Train Epoch: 5 [0/60000 (0%)]	Loss: 0.274164
Train Epoch: 5 [640/60000 (1%)]	Loss: 0.231656
Train Epoch: 5 [1280/60000 (2%)]	Loss: 0.221639
Train Epoch: 5 [1920/60000 (3%)]	Loss: 0.217534
Train Epoch: 5 [2560/60000 (4%)]	Loss: 0.134417
Train Epoch: 5 [3200/60000 (5%)]	Loss: 0.156011
Train Epoch: 5 [3840/60000 (6%)]	Loss: 0.201725
Train Epoch: 5 [4480/60000 (7%)]	Loss: 0.181683
Train Epoch: 5 [5120/60000 (9%)]	Loss: 0.127342
Train Epoch: 5 [5760/60000 (10%)]	Loss: 0.323498
Train Epoch: 5 [6400/60000 (11%)]	Loss: 0.226264
Train Epoch: 5 [7040/60000 (12%)]	Loss: 0.213340
Train Epoch: 5 [7680/60000 (13%)]	Loss: 0.225741
Train Epoch: 5 [8320/60000 (14%)]	Loss: 0.212258
Train Epoch: 5 [8960/60000 (15%)]	Loss: 0.220606
Train Epoch: 5 [9600/60000 (16%)]	Loss: 0.212915
Train Epoch: 5 [10240/60000 (17%)]	Loss: 0.270461
Train Epoch: 5 [10880/60000 (18%)]	Loss: 0.135657
Train Epoch: 5 [11520/60000 (19%)]	Loss: 0.223299
Train Epoch: 5 [12160/60000 (20%)]	Loss: 0.437413
Train Epoch: 5 [12800/60000 (21%)]	Loss: 0.475130
Train Epoch: 5 [13440/60000 (22%)]	Loss: 0.259950
Train Epoch: 5 [14080/60000 (23%)]	Loss: 0.121993
Train Epoch: 5 [14720/60000 (25%)]	Loss: 0.187580
Train Epoch: 5 [15360/60000 (26%)]	Loss: 0.208080
Train Epoch: 5 [16000/60000 (27%)]	Loss: 0.172820
Train Epoch: 5 [16640/60000 (28%)]	Loss: 0.197696
Train Epoch: 5 [17280/60000 (29%)]	Loss: 0.287385
Train Epoch: 5 [17920/60000 (30%)]	Loss: 0.140258
Train Epoch: 5 [18560/60000 (31%)]	Loss: 0.210366
Train Epoch: 5 [19200/60000 (32%)]	Loss: 0.281315
Train Epoch: 5 [19840/60000 (33%)]	Loss: 0.248378
Train Epoch: 5 [20480/60000 (34%)]	Loss: 0.237244
Train Epoch: 5 [21120/60000 (35%)]	Loss: 0.158226
Train Epoch: 5 [21760/60000 (36%)]	Loss: 0.184475
Train Epoch: 5 [22400/60000 (37%)]	Loss: 0.132852
Train Epoch: 5 [23040/60000 (38%)]	Loss: 0.211762
Train Epoch: 5 [23680/60000 (39%)]	Loss: 0.288887
Train Epoch: 5 [24320/60000 (41%)]	Loss: 0.400819
Train Epoch: 5 [24960/60000 (42%)]	Loss: 0.191266
Train Epoch: 5 [25600/60000 (43%)]	Loss: 0.118432
Train Epoch: 5 [26240/60000 (44%)]	Loss: 0.155194
Train Epoch: 5 [26880/60000 (45%)]	Loss: 0.218400
Train Epoch: 5 [27520/60000 (46%)]	Loss: 0.208711
Train Epoch: 5 [28160/60000 (47%)]	Loss: 0.157633
Train Epoch: 5 [28800/60000 (48%)]	Loss: 0.235771
Train Epoch: 5 [29440/60000 (49%)]	Loss: 0.282007
Train Epoch: 5 [30080/60000 (50%)]	Loss: 0.131590
Train Epoch: 5 [30720/60000 (51%)]	Loss: 0.296121
Train Epoch: 5 [31360/60000 (52%)]	Loss: 0.200366
Train Epoch: 5 [32000/60000 (53%)]	Loss: 0.207146
Train Epoch: 5 [32640/60000 (54%)]	Loss: 0.113057
Train Epoch: 5 [33280/60000 (55%)]	Loss: 0.241583
Train Epoch: 5 [33920/60000 (57%)]	Loss: 0.285096
Train Epoch: 5 [34560/60000 (58%)]	Loss: 0.280494
Train Epoch: 5 [35200/60000 (59%)]	Loss: 0.209738
Train Epoch: 5 [35840/60000 (60%)]	Loss: 0.173802
Train Epoch: 5 [36480/60000 (61%)]	Loss: 0.344032
Train Epoch: 5 [37120/60000 (62%)]	Loss: 0.159101
Train Epoch: 5 [37760/60000 (63%)]	Loss: 0.077197
Train Epoch: 5 [38400/60000 (64%)]	Loss: 0.252550
Train Epoch: 5 [39040/60000 (65%)]	Loss: 0.239048
Train Epoch: 5 [39680/60000 (66%)]	Loss: 0.101220
Train Epoch: 5 [40320/60000 (67%)]	Loss: 0.145567
Train Epoch: 5 [40960/60000 (68%)]	Loss: 0.047937
Train Epoch: 5 [41600/60000 (69%)]	Loss: 0.093983
Train Epoch: 5 [42240/60000 (70%)]	Loss: 0.413295
Train Epoch: 5 [42880/60000 (71%)]	Loss: 0.229182
Train Epoch: 5 [43520/60000 (72%)]	Loss: 0.436376
Train Epoch: 5 [44160/60000 (74%)]	Loss: 0.131919
Train Epoch: 5 [44800/60000 (75%)]	Loss: 0.410229
Train Epoch: 5 [45440/60000 (76%)]	Loss: 0.204255
Train Epoch: 5 [46080/60000 (77%)]	Loss: 0.217622
Train Epoch: 5 [46720/60000 (78%)]	Loss: 0.167461
Train Epoch: 5 [47360/60000 (79%)]	Loss: 0.171001
Train Epoch: 5 [48000/60000 (80%)]	Loss: 0.167110
Train Epoch: 5 [48640/60000 (81%)]	Loss: 0.223134
Train Epoch: 5 [49280/60000 (82%)]	Loss: 0.182766
Train Epoch: 5 [49920/60000 (83%)]	Loss: 0.175422
Train Epoch: 5 [50560/60000 (84%)]	Loss: 0.284434
Train Epoch: 5 [51200/60000 (85%)]	Loss: 0.195946
Train Epoch: 5 [51840/60000 (86%)]	Loss: 0.264567
Train Epoch: 5 [52480/60000 (87%)]	Loss: 0.332761
Train Epoch: 5 [53120/60000 (88%)]	Loss: 0.395759
Train Epoch: 5 [53760/60000 (90%)]	Loss: 0.115216
Train Epoch: 5 [54400/60000 (91%)]	Loss: 0.301400
Train Epoch: 5 [55040/60000 (92%)]	Loss: 0.071039
Train Epoch: 5 [55680/60000 (93%)]	Loss: 0.173142
Train Epoch: 5 [56320/60000 (94%)]	Loss: 0.104325
Train Epoch: 5 [56960/60000 (95%)]	Loss: 0.148404
Train Epoch: 5 [57600/60000 (96%)]	Loss: 0.097343
Train Epoch: 5 [58240/60000 (97%)]	Loss: 0.074098
Train Epoch: 5 [58880/60000 (98%)]	Loss: 0.251396
Train Epoch: 5 [59520/60000 (99%)]	Loss: 0.216123

Test set: Average loss: 0.0793, Accuracy: 9736/10000 (97%)

Train Epoch: 6 [0/60000 (0%)]	Loss: 0.166336
Train Epoch: 6 [640/60000 (1%)]	Loss: 0.177212
Train Epoch: 6 [1280/60000 (2%)]	Loss: 0.173440
Train Epoch: 6 [1920/60000 (3%)]	Loss: 0.278061
Train Epoch: 6 [2560/60000 (4%)]	Loss: 0.196190
Train Epoch: 6 [3200/60000 (5%)]	Loss: 0.195176
Train Epoch: 6 [3840/60000 (6%)]	Loss: 0.215426
Train Epoch: 6 [4480/60000 (7%)]	Loss: 0.395491
Train Epoch: 6 [5120/60000 (9%)]	Loss: 0.369025
Train Epoch: 6 [5760/60000 (10%)]	Loss: 0.092888
Train Epoch: 6 [6400/60000 (11%)]	Loss: 0.186412
Train Epoch: 6 [7040/60000 (12%)]	Loss: 0.401001
Train Epoch: 6 [7680/60000 (13%)]	Loss: 0.138095
Train Epoch: 6 [8320/60000 (14%)]	Loss: 0.105428
Train Epoch: 6 [8960/60000 (15%)]	Loss: 0.123254
Train Epoch: 6 [9600/60000 (16%)]	Loss: 0.211080
Train Epoch: 6 [10240/60000 (17%)]	Loss: 0.242884
Train Epoch: 6 [10880/60000 (18%)]	Loss: 0.293314
Train Epoch: 6 [11520/60000 (19%)]	Loss: 0.639213
Train Epoch: 6 [12160/60000 (20%)]	Loss: 0.494491
Train Epoch: 6 [12800/60000 (21%)]	Loss: 0.375642
Train Epoch: 6 [13440/60000 (22%)]	Loss: 0.168299
Train Epoch: 6 [14080/60000 (23%)]	Loss: 0.164487
Train Epoch: 6 [14720/60000 (25%)]	Loss: 0.315660
Train Epoch: 6 [15360/60000 (26%)]	Loss: 0.362971
Train Epoch: 6 [16000/60000 (27%)]	Loss: 0.270098
Train Epoch: 6 [16640/60000 (28%)]	Loss: 0.184845
Train Epoch: 6 [17280/60000 (29%)]	Loss: 0.098442
Train Epoch: 6 [17920/60000 (30%)]	Loss: 0.176058
Train Epoch: 6 [18560/60000 (31%)]	Loss: 0.307367
Train Epoch: 6 [19200/60000 (32%)]	Loss: 0.246186
Train Epoch: 6 [19840/60000 (33%)]	Loss: 0.323171
Train Epoch: 6 [20480/60000 (34%)]	Loss: 0.280123
Train Epoch: 6 [21120/60000 (35%)]	Loss: 0.287222
Train Epoch: 6 [21760/60000 (36%)]	Loss: 0.158834
Train Epoch: 6 [22400/60000 (37%)]	Loss: 0.170442
Train Epoch: 6 [23040/60000 (38%)]	Loss: 0.446494
Train Epoch: 6 [23680/60000 (39%)]	Loss: 0.167727
Train Epoch: 6 [24320/60000 (41%)]	Loss: 0.253836
Train Epoch: 6 [24960/60000 (42%)]	Loss: 0.154910
Train Epoch: 6 [25600/60000 (43%)]	Loss: 0.297387
Train Epoch: 6 [26240/60000 (44%)]	Loss: 0.171043
Train Epoch: 6 [26880/60000 (45%)]	Loss: 0.184092
Train Epoch: 6 [27520/60000 (46%)]	Loss: 0.173465
Train Epoch: 6 [28160/60000 (47%)]	Loss: 0.417981
Train Epoch: 6 [28800/60000 (48%)]	Loss: 0.681483
Train Epoch: 6 [29440/60000 (49%)]	Loss: 0.151250
Train Epoch: 6 [30080/60000 (50%)]	Loss: 0.389995
Train Epoch: 6 [30720/60000 (51%)]	Loss: 0.159561
Train Epoch: 6 [31360/60000 (52%)]	Loss: 0.312322
Train Epoch: 6 [32000/60000 (53%)]	Loss: 0.277379
Train Epoch: 6 [32640/60000 (54%)]	Loss: 0.143581
Train Epoch: 6 [33280/60000 (55%)]	Loss: 0.335243
Train Epoch: 6 [33920/60000 (57%)]	Loss: 0.119857
Train Epoch: 6 [34560/60000 (58%)]	Loss: 0.143197
Train Epoch: 6 [35200/60000 (59%)]	Loss: 0.239978
Train Epoch: 6 [35840/60000 (60%)]	Loss: 0.295368
Train Epoch: 6 [36480/60000 (61%)]	Loss: 0.220212
Train Epoch: 6 [37120/60000 (62%)]	Loss: 0.096295
Train Epoch: 6 [37760/60000 (63%)]	Loss: 0.306999
Train Epoch: 6 [38400/60000 (64%)]	Loss: 0.167993
Train Epoch: 6 [39040/60000 (65%)]	Loss: 0.095115
Train Epoch: 6 [39680/60000 (66%)]	Loss: 0.149591
Train Epoch: 6 [40320/60000 (67%)]	Loss: 0.184489
Train Epoch: 6 [40960/60000 (68%)]	Loss: 0.154977
Train Epoch: 6 [41600/60000 (69%)]	Loss: 0.109101
Train Epoch: 6 [42240/60000 (70%)]	Loss: 0.306995
Train Epoch: 6 [42880/60000 (71%)]	Loss: 0.150501
Train Epoch: 6 [43520/60000 (72%)]	Loss: 0.107667
Train Epoch: 6 [44160/60000 (74%)]	Loss: 0.196727
Train Epoch: 6 [44800/60000 (75%)]	Loss: 0.209088
Train Epoch: 6 [45440/60000 (76%)]	Loss: 0.155056
Train Epoch: 6 [46080/60000 (77%)]	Loss: 0.190872
Train Epoch: 6 [46720/60000 (78%)]	Loss: 0.161997
Train Epoch: 6 [47360/60000 (79%)]	Loss: 0.120235
Train Epoch: 6 [48000/60000 (80%)]	Loss: 0.567525
Train Epoch: 6 [48640/60000 (81%)]	Loss: 0.357469
Train Epoch: 6 [49280/60000 (82%)]	Loss: 0.084540
Train Epoch: 6 [49920/60000 (83%)]	Loss: 0.289413
Train Epoch: 6 [50560/60000 (84%)]	Loss: 0.115442
Train Epoch: 6 [51200/60000 (85%)]	Loss: 0.177020
Train Epoch: 6 [51840/60000 (86%)]	Loss: 0.297659
Train Epoch: 6 [52480/60000 (87%)]	Loss: 0.151834
Train Epoch: 6 [53120/60000 (88%)]	Loss: 0.237470
Train Epoch: 6 [53760/60000 (90%)]	Loss: 0.206151
Train Epoch: 6 [54400/60000 (91%)]	Loss: 0.162496
Train Epoch: 6 [55040/60000 (92%)]	Loss: 0.118576
Train Epoch: 6 [55680/60000 (93%)]	Loss: 0.179058
Train Epoch: 6 [56320/60000 (94%)]	Loss: 0.494467
Train Epoch: 6 [56960/60000 (95%)]	Loss: 0.157796
Train Epoch: 6 [57600/60000 (96%)]	Loss: 0.123645
Train Epoch: 6 [58240/60000 (97%)]	Loss: 0.216377
Train Epoch: 6 [58880/60000 (98%)]	Loss: 0.195910
Train Epoch: 6 [59520/60000 (99%)]	Loss: 0.127426

Test set: Average loss: 0.0691, Accuracy: 9779/10000 (98%)

Train Epoch: 7 [0/60000 (0%)]	Loss: 0.238695
Train Epoch: 7 [640/60000 (1%)]	Loss: 0.069180
Train Epoch: 7 [1280/60000 (2%)]	Loss: 0.182545
Train Epoch: 7 [1920/60000 (3%)]	Loss: 0.264609
Train Epoch: 7 [2560/60000 (4%)]	Loss: 0.242664
Train Epoch: 7 [3200/60000 (5%)]	Loss: 0.118079
Train Epoch: 7 [3840/60000 (6%)]	Loss: 0.152568
Train Epoch: 7 [4480/60000 (7%)]	Loss: 0.136664
Train Epoch: 7 [5120/60000 (9%)]	Loss: 0.189271
Train Epoch: 7 [5760/60000 (10%)]	Loss: 0.259070
Train Epoch: 7 [6400/60000 (11%)]	Loss: 0.270009
Train Epoch: 7 [7040/60000 (12%)]	Loss: 0.203359
Train Epoch: 7 [7680/60000 (13%)]	Loss: 0.213923
Train Epoch: 7 [8320/60000 (14%)]	Loss: 0.324505
Train Epoch: 7 [8960/60000 (15%)]	Loss: 0.258868
Train Epoch: 7 [9600/60000 (16%)]	Loss: 0.157469
Train Epoch: 7 [10240/60000 (17%)]	Loss: 0.584216
Train Epoch: 7 [10880/60000 (18%)]	Loss: 0.261341
Train Epoch: 7 [11520/60000 (19%)]	Loss: 0.193367
Train Epoch: 7 [12160/60000 (20%)]	Loss: 0.163610
Train Epoch: 7 [12800/60000 (21%)]	Loss: 0.149796
Train Epoch: 7 [13440/60000 (22%)]	Loss: 0.116201
Train Epoch: 7 [14080/60000 (23%)]	Loss: 0.213471
Train Epoch: 7 [14720/60000 (25%)]	Loss: 0.224344
Train Epoch: 7 [15360/60000 (26%)]	Loss: 0.118353
Train Epoch: 7 [16000/60000 (27%)]	Loss: 0.272746
Train Epoch: 7 [16640/60000 (28%)]	Loss: 0.141830
Train Epoch: 7 [17280/60000 (29%)]	Loss: 0.158590
Train Epoch: 7 [17920/60000 (30%)]	Loss: 0.452490
Train Epoch: 7 [18560/60000 (31%)]	Loss: 0.211903
Train Epoch: 7 [19200/60000 (32%)]	Loss: 0.270624
Train Epoch: 7 [19840/60000 (33%)]	Loss: 0.336491
Train Epoch: 7 [20480/60000 (34%)]	Loss: 0.347319
Train Epoch: 7 [21120/60000 (35%)]	Loss: 0.301990
Train Epoch: 7 [21760/60000 (36%)]	Loss: 0.343419
Train Epoch: 7 [22400/60000 (37%)]	Loss: 0.128868
Train Epoch: 7 [23040/60000 (38%)]	Loss: 0.417029
Train Epoch: 7 [23680/60000 (39%)]	Loss: 0.279091
Train Epoch: 7 [24320/60000 (41%)]	Loss: 0.400411
Train Epoch: 7 [24960/60000 (42%)]	Loss: 0.133842
Train Epoch: 7 [25600/60000 (43%)]	Loss: 0.205805
Train Epoch: 7 [26240/60000 (44%)]	Loss: 0.211226
Train Epoch: 7 [26880/60000 (45%)]	Loss: 0.093308
Train Epoch: 7 [27520/60000 (46%)]	Loss: 0.201403
Train Epoch: 7 [28160/60000 (47%)]	Loss: 0.092497
Train Epoch: 7 [28800/60000 (48%)]	Loss: 0.183695
Train Epoch: 7 [29440/60000 (49%)]	Loss: 0.203633
Train Epoch: 7 [30080/60000 (50%)]	Loss: 0.257370
Train Epoch: 7 [30720/60000 (51%)]	Loss: 0.254085
Train Epoch: 7 [31360/60000 (52%)]	Loss: 0.129679
Train Epoch: 7 [32000/60000 (53%)]	Loss: 0.143966
Train Epoch: 7 [32640/60000 (54%)]	Loss: 0.336283
Train Epoch: 7 [33280/60000 (55%)]	Loss: 0.136440
Train Epoch: 7 [33920/60000 (57%)]	Loss: 0.192106
Train Epoch: 7 [34560/60000 (58%)]	Loss: 0.323964
Train Epoch: 7 [35200/60000 (59%)]	Loss: 0.121070
Train Epoch: 7 [35840/60000 (60%)]	Loss: 0.271260
Train Epoch: 7 [36480/60000 (61%)]	Loss: 0.298934
Train Epoch: 7 [37120/60000 (62%)]	Loss: 0.185636
Train Epoch: 7 [37760/60000 (63%)]	Loss: 0.193501
Train Epoch: 7 [38400/60000 (64%)]	Loss: 0.239755
Train Epoch: 7 [39040/60000 (65%)]	Loss: 0.231702
Train Epoch: 7 [39680/60000 (66%)]	Loss: 0.136329
Train Epoch: 7 [40320/60000 (67%)]	Loss: 0.183596
Train Epoch: 7 [40960/60000 (68%)]	Loss: 0.145942
Train Epoch: 7 [41600/60000 (69%)]	Loss: 0.120262
Train Epoch: 7 [42240/60000 (70%)]	Loss: 0.229096
Train Epoch: 7 [42880/60000 (71%)]	Loss: 0.339243
Train Epoch: 7 [43520/60000 (72%)]	Loss: 0.291523
Train Epoch: 7 [44160/60000 (74%)]	Loss: 0.163675
Train Epoch: 7 [44800/60000 (75%)]	Loss: 0.341528
Train Epoch: 7 [45440/60000 (76%)]	Loss: 0.193145
Train Epoch: 7 [46080/60000 (77%)]	Loss: 0.275938
Train Epoch: 7 [46720/60000 (78%)]	Loss: 0.089451
Train Epoch: 7 [47360/60000 (79%)]	Loss: 0.186925
Train Epoch: 7 [48000/60000 (80%)]	Loss: 0.122016
Train Epoch: 7 [48640/60000 (81%)]	Loss: 0.116524
Train Epoch: 7 [49280/60000 (82%)]	Loss: 0.166615
Train Epoch: 7 [49920/60000 (83%)]	Loss: 0.203388
Train Epoch: 7 [50560/60000 (84%)]	Loss: 0.136161
Train Epoch: 7 [51200/60000 (85%)]	Loss: 0.167379
Train Epoch: 7 [51840/60000 (86%)]	Loss: 0.270751
Train Epoch: 7 [52480/60000 (87%)]	Loss: 0.385404
Train Epoch: 7 [53120/60000 (88%)]	Loss: 0.329275
Train Epoch: 7 [53760/60000 (90%)]	Loss: 0.113516
Train Epoch: 7 [54400/60000 (91%)]	Loss: 0.316852
Train Epoch: 7 [55040/60000 (92%)]	Loss: 0.147377
Train Epoch: 7 [55680/60000 (93%)]	Loss: 0.212370
Train Epoch: 7 [56320/60000 (94%)]	Loss: 0.203172
Train Epoch: 7 [56960/60000 (95%)]	Loss: 0.242213
Train Epoch: 7 [57600/60000 (96%)]	Loss: 0.434109
Train Epoch: 7 [58240/60000 (97%)]	Loss: 0.180775
Train Epoch: 7 [58880/60000 (98%)]	Loss: 0.118768
Train Epoch: 7 [59520/60000 (99%)]	Loss: 0.310509

Test set: Average loss: 0.0687, Accuracy: 9774/10000 (98%)

Train Epoch: 8 [0/60000 (0%)]	Loss: 0.400091
Train Epoch: 8 [640/60000 (1%)]	Loss: 0.152526
Train Epoch: 8 [1280/60000 (2%)]	Loss: 0.191972
Train Epoch: 8 [1920/60000 (3%)]	Loss: 0.295449
Train Epoch: 8 [2560/60000 (4%)]	Loss: 0.181650
Train Epoch: 8 [3200/60000 (5%)]	Loss: 0.370180
Train Epoch: 8 [3840/60000 (6%)]	Loss: 0.332610
Train Epoch: 8 [4480/60000 (7%)]	Loss: 0.213501
Train Epoch: 8 [5120/60000 (9%)]	Loss: 0.241246
Train Epoch: 8 [5760/60000 (10%)]	Loss: 0.381942
Train Epoch: 8 [6400/60000 (11%)]	Loss: 0.097483
Train Epoch: 8 [7040/60000 (12%)]	Loss: 0.179237
Train Epoch: 8 [7680/60000 (13%)]	Loss: 0.229303
Train Epoch: 8 [8320/60000 (14%)]	Loss: 0.220130
Train Epoch: 8 [8960/60000 (15%)]	Loss: 0.097682
Train Epoch: 8 [9600/60000 (16%)]	Loss: 0.146019
Train Epoch: 8 [10240/60000 (17%)]	Loss: 0.164624
Train Epoch: 8 [10880/60000 (18%)]	Loss: 0.119153
Train Epoch: 8 [11520/60000 (19%)]	Loss: 0.212751
Train Epoch: 8 [12160/60000 (20%)]	Loss: 0.218877
Train Epoch: 8 [12800/60000 (21%)]	Loss: 0.398361
Train Epoch: 8 [13440/60000 (22%)]	Loss: 0.141618
Train Epoch: 8 [14080/60000 (23%)]	Loss: 0.198022
Train Epoch: 8 [14720/60000 (25%)]	Loss: 0.145388
Train Epoch: 8 [15360/60000 (26%)]	Loss: 0.200622
Train Epoch: 8 [16000/60000 (27%)]	Loss: 0.202905
Train Epoch: 8 [16640/60000 (28%)]	Loss: 0.089408
Train Epoch: 8 [17280/60000 (29%)]	Loss: 0.092478
Train Epoch: 8 [17920/60000 (30%)]	Loss: 0.150676
Train Epoch: 8 [18560/60000 (31%)]	Loss: 0.273498
Train Epoch: 8 [19200/60000 (32%)]	Loss: 0.153092
Train Epoch: 8 [19840/60000 (33%)]	Loss: 0.187569
Train Epoch: 8 [20480/60000 (34%)]	Loss: 0.168195
Train Epoch: 8 [21120/60000 (35%)]	Loss: 0.182713
Train Epoch: 8 [21760/60000 (36%)]	Loss: 0.121985
Train Epoch: 8 [22400/60000 (37%)]	Loss: 0.125071
Train Epoch: 8 [23040/60000 (38%)]	Loss: 0.342869
Train Epoch: 8 [23680/60000 (39%)]	Loss: 0.141305
Train Epoch: 8 [24320/60000 (41%)]	Loss: 0.160070
Train Epoch: 8 [24960/60000 (42%)]	Loss: 0.393508
Train Epoch: 8 [25600/60000 (43%)]	Loss: 0.208634
Train Epoch: 8 [26240/60000 (44%)]	Loss: 0.118412
Train Epoch: 8 [26880/60000 (45%)]	Loss: 0.087676
Train Epoch: 8 [27520/60000 (46%)]	Loss: 0.192088
Train Epoch: 8 [28160/60000 (47%)]	Loss: 0.292131
Train Epoch: 8 [28800/60000 (48%)]	Loss: 0.349722
Train Epoch: 8 [29440/60000 (49%)]	Loss: 0.209598
Train Epoch: 8 [30080/60000 (50%)]	Loss: 0.246638
Train Epoch: 8 [30720/60000 (51%)]	Loss: 0.135610
Train Epoch: 8 [31360/60000 (52%)]	Loss: 0.291379
Train Epoch: 8 [32000/60000 (53%)]	Loss: 0.177169
Train Epoch: 8 [32640/60000 (54%)]	Loss: 0.119931
Train Epoch: 8 [33280/60000 (55%)]	Loss: 0.061965
Train Epoch: 8 [33920/60000 (57%)]	Loss: 0.222739
Train Epoch: 8 [34560/60000 (58%)]	Loss: 0.289190
Train Epoch: 8 [35200/60000 (59%)]	Loss: 0.112155
Train Epoch: 8 [35840/60000 (60%)]	Loss: 0.229224
Train Epoch: 8 [36480/60000 (61%)]	Loss: 0.362608
Train Epoch: 8 [37120/60000 (62%)]	Loss: 0.138838
Train Epoch: 8 [37760/60000 (63%)]	Loss: 0.229829
Train Epoch: 8 [38400/60000 (64%)]	Loss: 0.201637
Train Epoch: 8 [39040/60000 (65%)]	Loss: 0.114230
Train Epoch: 8 [39680/60000 (66%)]	Loss: 0.145208
Train Epoch: 8 [40320/60000 (67%)]	Loss: 0.172777
Train Epoch: 8 [40960/60000 (68%)]	Loss: 0.121140
Train Epoch: 8 [41600/60000 (69%)]	Loss: 0.174082
Train Epoch: 8 [42240/60000 (70%)]	Loss: 0.260587
Train Epoch: 8 [42880/60000 (71%)]	Loss: 0.273963
Train Epoch: 8 [43520/60000 (72%)]	Loss: 0.284604
Train Epoch: 8 [44160/60000 (74%)]	Loss: 0.161078
Train Epoch: 8 [44800/60000 (75%)]	Loss: 0.130752
Train Epoch: 8 [45440/60000 (76%)]	Loss: 0.246799
Train Epoch: 8 [46080/60000 (77%)]	Loss: 0.263351
Train Epoch: 8 [46720/60000 (78%)]	Loss: 0.107764
Train Epoch: 8 [47360/60000 (79%)]	Loss: 0.045004
Train Epoch: 8 [48000/60000 (80%)]	Loss: 0.122512
Train Epoch: 8 [48640/60000 (81%)]	Loss: 0.220129
Train Epoch: 8 [49280/60000 (82%)]	Loss: 0.099767
Train Epoch: 8 [49920/60000 (83%)]	Loss: 0.289344
Train Epoch: 8 [50560/60000 (84%)]	Loss: 0.115881
Train Epoch: 8 [51200/60000 (85%)]	Loss: 0.140058
Train Epoch: 8 [51840/60000 (86%)]	Loss: 0.183398
Train Epoch: 8 [52480/60000 (87%)]	Loss: 0.176359
Train Epoch: 8 [53120/60000 (88%)]	Loss: 0.129470
Train Epoch: 8 [53760/60000 (90%)]	Loss: 0.272180
Train Epoch: 8 [54400/60000 (91%)]	Loss: 0.167415
Train Epoch: 8 [55040/60000 (92%)]	Loss: 0.194334
Train Epoch: 8 [55680/60000 (93%)]	Loss: 0.297763
Train Epoch: 8 [56320/60000 (94%)]	Loss: 0.192485
Train Epoch: 8 [56960/60000 (95%)]	Loss: 0.237093
Train Epoch: 8 [57600/60000 (96%)]	Loss: 0.142890
Train Epoch: 8 [58240/60000 (97%)]	Loss: 0.160970
Train Epoch: 8 [58880/60000 (98%)]	Loss: 0.220537
Train Epoch: 8 [59520/60000 (99%)]	Loss: 0.377333

Test set: Average loss: 0.0624, Accuracy: 9796/10000 (98%)

Train Epoch: 9 [0/60000 (0%)]	Loss: 0.155629
Train Epoch: 9 [640/60000 (1%)]	Loss: 0.180203
Train Epoch: 9 [1280/60000 (2%)]	Loss: 0.165649
Train Epoch: 9 [1920/60000 (3%)]	Loss: 0.094051
Train Epoch: 9 [2560/60000 (4%)]	Loss: 0.219419
Train Epoch: 9 [3200/60000 (5%)]	Loss: 0.280034
Train Epoch: 9 [3840/60000 (6%)]	Loss: 0.209159
Train Epoch: 9 [4480/60000 (7%)]	Loss: 0.154131
Train Epoch: 9 [5120/60000 (9%)]	Loss: 0.104335
Train Epoch: 9 [5760/60000 (10%)]	Loss: 0.135716
Train Epoch: 9 [6400/60000 (11%)]	Loss: 0.140966
Train Epoch: 9 [7040/60000 (12%)]	Loss: 0.171098
Train Epoch: 9 [7680/60000 (13%)]	Loss: 0.201777
Train Epoch: 9 [8320/60000 (14%)]	Loss: 0.278955
Train Epoch: 9 [8960/60000 (15%)]	Loss: 0.134633
Train Epoch: 9 [9600/60000 (16%)]	Loss: 0.086113
Train Epoch: 9 [10240/60000 (17%)]	Loss: 0.275087
Train Epoch: 9 [10880/60000 (18%)]	Loss: 0.202018
Train Epoch: 9 [11520/60000 (19%)]	Loss: 0.456039
Train Epoch: 9 [12160/60000 (20%)]	Loss: 0.270068
Train Epoch: 9 [12800/60000 (21%)]	Loss: 0.201579
Train Epoch: 9 [13440/60000 (22%)]	Loss: 0.357491
Train Epoch: 9 [14080/60000 (23%)]	Loss: 0.159846
Train Epoch: 9 [14720/60000 (25%)]	Loss: 0.186796
Train Epoch: 9 [15360/60000 (26%)]	Loss: 0.075998
Train Epoch: 9 [16000/60000 (27%)]	Loss: 0.066529
Train Epoch: 9 [16640/60000 (28%)]	Loss: 0.112582
Train Epoch: 9 [17280/60000 (29%)]	Loss: 0.227846
Train Epoch: 9 [17920/60000 (30%)]	Loss: 0.218067
Train Epoch: 9 [18560/60000 (31%)]	Loss: 0.227501
Train Epoch: 9 [19200/60000 (32%)]	Loss: 0.297330
Train Epoch: 9 [19840/60000 (33%)]	Loss: 0.249750
Train Epoch: 9 [20480/60000 (34%)]	Loss: 0.126650
Train Epoch: 9 [21120/60000 (35%)]	Loss: 0.235015
Train Epoch: 9 [21760/60000 (36%)]	Loss: 0.191965
Train Epoch: 9 [22400/60000 (37%)]	Loss: 0.206619
Train Epoch: 9 [23040/60000 (38%)]	Loss: 0.104831
Train Epoch: 9 [23680/60000 (39%)]	Loss: 0.110880
Train Epoch: 9 [24320/60000 (41%)]	Loss: 0.347205
Train Epoch: 9 [24960/60000 (42%)]	Loss: 0.276492
Train Epoch: 9 [25600/60000 (43%)]	Loss: 0.065573
Train Epoch: 9 [26240/60000 (44%)]	Loss: 0.129329
Train Epoch: 9 [26880/60000 (45%)]	Loss: 0.222425
Train Epoch: 9 [27520/60000 (46%)]	Loss: 0.253131
Train Epoch: 9 [28160/60000 (47%)]	Loss: 0.100319
Train Epoch: 9 [28800/60000 (48%)]	Loss: 0.246236
Train Epoch: 9 [29440/60000 (49%)]	Loss: 0.253961
Train Epoch: 9 [30080/60000 (50%)]	Loss: 0.252975
Train Epoch: 9 [30720/60000 (51%)]	Loss: 0.045665
Train Epoch: 9 [31360/60000 (52%)]	Loss: 0.110188
Train Epoch: 9 [32000/60000 (53%)]	Loss: 0.052142
Train Epoch: 9 [32640/60000 (54%)]	Loss: 0.216492
Train Epoch: 9 [33280/60000 (55%)]	Loss: 0.232279
Train Epoch: 9 [33920/60000 (57%)]	Loss: 0.450011
Train Epoch: 9 [34560/60000 (58%)]	Loss: 0.483262
Train Epoch: 9 [35200/60000 (59%)]	Loss: 0.178879
Train Epoch: 9 [35840/60000 (60%)]	Loss: 0.105483
Train Epoch: 9 [36480/60000 (61%)]	Loss: 0.202518
Train Epoch: 9 [37120/60000 (62%)]	Loss: 0.274042
Train Epoch: 9 [37760/60000 (63%)]	Loss: 0.184607
Train Epoch: 9 [38400/60000 (64%)]	Loss: 0.281977
Train Epoch: 9 [39040/60000 (65%)]	Loss: 0.076208
Train Epoch: 9 [39680/60000 (66%)]	Loss: 0.248072
Train Epoch: 9 [40320/60000 (67%)]	Loss: 0.193254
Train Epoch: 9 [40960/60000 (68%)]	Loss: 0.216062
Train Epoch: 9 [41600/60000 (69%)]	Loss: 0.243379
Train Epoch: 9 [42240/60000 (70%)]	Loss: 0.204963
Train Epoch: 9 [42880/60000 (71%)]	Loss: 0.213194
Train Epoch: 9 [43520/60000 (72%)]	Loss: 0.153007
Train Epoch: 9 [44160/60000 (74%)]	Loss: 0.079691
Train Epoch: 9 [44800/60000 (75%)]	Loss: 0.359319
Train Epoch: 9 [45440/60000 (76%)]	Loss: 0.091213
Train Epoch: 9 [46080/60000 (77%)]	Loss: 0.236308
Train Epoch: 9 [46720/60000 (78%)]	Loss: 0.163690
Train Epoch: 9 [47360/60000 (79%)]	Loss: 0.288776
Train Epoch: 9 [48000/60000 (80%)]	Loss: 0.067516
Train Epoch: 9 [48640/60000 (81%)]	Loss: 0.214905
Train Epoch: 9 [49280/60000 (82%)]	Loss: 0.068250
Train Epoch: 9 [49920/60000 (83%)]	Loss: 0.166835
Train Epoch: 9 [50560/60000 (84%)]	Loss: 0.246616
Train Epoch: 9 [51200/60000 (85%)]	Loss: 0.259778
Train Epoch: 9 [51840/60000 (86%)]	Loss: 0.107666
Train Epoch: 9 [52480/60000 (87%)]	Loss: 0.147197
Train Epoch: 9 [53120/60000 (88%)]	Loss: 0.094426
Train Epoch: 9 [53760/60000 (90%)]	Loss: 0.295506
Train Epoch: 9 [54400/60000 (91%)]	Loss: 0.200240
Train Epoch: 9 [55040/60000 (92%)]	Loss: 0.329342
Train Epoch: 9 [55680/60000 (93%)]	Loss: 0.288517
Train Epoch: 9 [56320/60000 (94%)]	Loss: 0.200540
Train Epoch: 9 [56960/60000 (95%)]	Loss: 0.123969
Train Epoch: 9 [57600/60000 (96%)]	Loss: 0.215670
Train Epoch: 9 [58240/60000 (97%)]	Loss: 0.056512
Train Epoch: 9 [58880/60000 (98%)]	Loss: 0.102123
Train Epoch: 9 [59520/60000 (99%)]	Loss: 0.167506

Test set: Average loss: 0.0572, Accuracy: 9807/10000 (98%)

Train Epoch: 10 [0/60000 (0%)]	Loss: 0.063941
Train Epoch: 10 [640/60000 (1%)]	Loss: 0.151969
Train Epoch: 10 [1280/60000 (2%)]	Loss: 0.101176
Train Epoch: 10 [1920/60000 (3%)]	Loss: 0.334071
Train Epoch: 10 [2560/60000 (4%)]	Loss: 0.148542
Train Epoch: 10 [3200/60000 (5%)]	Loss: 0.104701
Train Epoch: 10 [3840/60000 (6%)]	Loss: 0.101654
Train Epoch: 10 [4480/60000 (7%)]	Loss: 0.226924
Train Epoch: 10 [5120/60000 (9%)]	Loss: 0.649824
Train Epoch: 10 [5760/60000 (10%)]	Loss: 0.152952
Train Epoch: 10 [6400/60000 (11%)]	Loss: 0.308312
Train Epoch: 10 [7040/60000 (12%)]	Loss: 0.309165
Train Epoch: 10 [7680/60000 (13%)]	Loss: 0.079549
Train Epoch: 10 [8320/60000 (14%)]	Loss: 0.148520
Train Epoch: 10 [8960/60000 (15%)]	Loss: 0.184744
Train Epoch: 10 [9600/60000 (16%)]	Loss: 0.307692
Train Epoch: 10 [10240/60000 (17%)]	Loss: 0.175493
Train Epoch: 10 [10880/60000 (18%)]	Loss: 0.223569
Train Epoch: 10 [11520/60000 (19%)]	Loss: 0.251596
Train Epoch: 10 [12160/60000 (20%)]	Loss: 0.106342
Train Epoch: 10 [12800/60000 (21%)]	Loss: 0.303456
Train Epoch: 10 [13440/60000 (22%)]	Loss: 0.200586
Train Epoch: 10 [14080/60000 (23%)]	Loss: 0.196428
Train Epoch: 10 [14720/60000 (25%)]	Loss: 0.398756
Train Epoch: 10 [15360/60000 (26%)]	Loss: 0.111398
Train Epoch: 10 [16000/60000 (27%)]	Loss: 0.144420
Train Epoch: 10 [16640/60000 (28%)]	Loss: 0.144085
Train Epoch: 10 [17280/60000 (29%)]	Loss: 0.287735
Train Epoch: 10 [17920/60000 (30%)]	Loss: 0.217186
Train Epoch: 10 [18560/60000 (31%)]	Loss: 0.272109
Train Epoch: 10 [19200/60000 (32%)]	Loss: 0.212966
Train Epoch: 10 [19840/60000 (33%)]	Loss: 0.483079
Train Epoch: 10 [20480/60000 (34%)]	Loss: 0.110542
Train Epoch: 10 [21120/60000 (35%)]	Loss: 0.140796
Train Epoch: 10 [21760/60000 (36%)]	Loss: 0.073781
Train Epoch: 10 [22400/60000 (37%)]	Loss: 0.179443
Train Epoch: 10 [23040/60000 (38%)]	Loss: 0.229120
Train Epoch: 10 [23680/60000 (39%)]	Loss: 0.124185
Train Epoch: 10 [24320/60000 (41%)]	Loss: 0.078742
Train Epoch: 10 [24960/60000 (42%)]	Loss: 0.270781
Train Epoch: 10 [25600/60000 (43%)]	Loss: 0.316126
Train Epoch: 10 [26240/60000 (44%)]	Loss: 0.194335
Train Epoch: 10 [26880/60000 (45%)]	Loss: 0.060066
Train Epoch: 10 [27520/60000 (46%)]	Loss: 0.104825
Train Epoch: 10 [28160/60000 (47%)]	Loss: 0.064073
Train Epoch: 10 [28800/60000 (48%)]	Loss: 0.161329
Train Epoch: 10 [29440/60000 (49%)]	Loss: 0.196067
Train Epoch: 10 [30080/60000 (50%)]	Loss: 0.136898
Train Epoch: 10 [30720/60000 (51%)]	Loss: 0.204393
Train Epoch: 10 [31360/60000 (52%)]	Loss: 0.146195
Train Epoch: 10 [32000/60000 (53%)]	Loss: 0.158654
Train Epoch: 10 [32640/60000 (54%)]	Loss: 0.095445
Train Epoch: 10 [33280/60000 (55%)]	Loss: 0.274451
Train Epoch: 10 [33920/60000 (57%)]	Loss: 0.433168
Train Epoch: 10 [34560/60000 (58%)]	Loss: 0.112105
Train Epoch: 10 [35200/60000 (59%)]	Loss: 0.123602
Train Epoch: 10 [35840/60000 (60%)]	Loss: 0.482569
Train Epoch: 10 [36480/60000 (61%)]	Loss: 0.147898
Train Epoch: 10 [37120/60000 (62%)]	Loss: 0.295110
Train Epoch: 10 [37760/60000 (63%)]	Loss: 0.267862
Train Epoch: 10 [38400/60000 (64%)]	Loss: 0.094174
Train Epoch: 10 [39040/60000 (65%)]	Loss: 0.432612
Train Epoch: 10 [39680/60000 (66%)]	Loss: 0.216504
Train Epoch: 10 [40320/60000 (67%)]	Loss: 0.175994
Train Epoch: 10 [40960/60000 (68%)]	Loss: 0.165823
Train Epoch: 10 [41600/60000 (69%)]	Loss: 0.223522
Train Epoch: 10 [42240/60000 (70%)]	Loss: 0.159783
Train Epoch: 10 [42880/60000 (71%)]	Loss: 0.222450
Train Epoch: 10 [43520/60000 (72%)]	Loss: 0.144114
Train Epoch: 10 [44160/60000 (74%)]	Loss: 0.158731
Train Epoch: 10 [44800/60000 (75%)]	Loss: 0.089033
Train Epoch: 10 [45440/60000 (76%)]	Loss: 0.165889
Train Epoch: 10 [46080/60000 (77%)]	Loss: 0.278269
Train Epoch: 10 [46720/60000 (78%)]	Loss: 0.253306
Train Epoch: 10 [47360/60000 (79%)]	Loss: 0.313436
Train Epoch: 10 [48000/60000 (80%)]	Loss: 0.120106
Train Epoch: 10 [48640/60000 (81%)]	Loss: 0.118673
Train Epoch: 10 [49280/60000 (82%)]	Loss: 0.164198
Train Epoch: 10 [49920/60000 (83%)]	Loss: 0.072301
Train Epoch: 10 [50560/60000 (84%)]	Loss: 0.116073
Train Epoch: 10 [51200/60000 (85%)]	Loss: 0.231153
Train Epoch: 10 [51840/60000 (86%)]	Loss: 0.123656
Train Epoch: 10 [52480/60000 (87%)]	Loss: 0.075421
Train Epoch: 10 [53120/60000 (88%)]	Loss: 0.130102
Train Epoch: 10 [53760/60000 (90%)]	Loss: 0.248063
Train Epoch: 10 [54400/60000 (91%)]	Loss: 0.201713
Train Epoch: 10 [55040/60000 (92%)]	Loss: 0.165778
Train Epoch: 10 [55680/60000 (93%)]	Loss: 0.219161
Train Epoch: 10 [56320/60000 (94%)]	Loss: 0.122359
Train Epoch: 10 [56960/60000 (95%)]	Loss: 0.132655
Train Epoch: 10 [57600/60000 (96%)]	Loss: 0.150823
Train Epoch: 10 [58240/60000 (97%)]	Loss: 0.111038
Train Epoch: 10 [58880/60000 (98%)]	Loss: 0.284887
Train Epoch: 10 [59520/60000 (99%)]	Loss: 0.081290

Test set: Average loss: 0.0545, Accuracy: 9817/10000 (98%)


real	1m14.587s
user	1m35.344s
sys	0m14.004s


```

# 6. Conclusion

This fun project will give you your own environment with a clean separation based on Docker. I want to leave you with a few ideas to take this to the next level:

1. Build the ```mxnet``` and ```tensorflow``` images.
2. Add your examples to a local volume on the host and use docker volumes to map the local example to any container.
3. Expose Jupyter notebook on your container, launch it in your entrypoint.sh and ```nvidia-docker run``` your images with a jupyter port (```8888```) binding between the host and the container.


[1]:	https://www.nvidia.com/en-us/data-center/tesla-v100/
[2]:	https://aws.amazon.com/about-aws/whats-new/2017/10/introducing-amazon-ec2-p3-instances/
[3]:    https://aws.amazon.com/ecr/
[4]:    https://aws.amazon.com/marketplace/pp/B01JBL2M0O
[5]:    #
[6]:    #
[7]:    #
[8]:    http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/user-data.html
[9]:    #
[10]:   #
