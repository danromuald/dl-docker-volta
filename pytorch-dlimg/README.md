## Running PyTorch and Anaconda3 on Volta on Amazon EC2

# 1. Introduction

PyTorch
Volta
Cuda9

# 2. Environment and Prerequisites

AWS P3
Ubuntu image

# 3. Supported Compute Capabilities

AWS P2

# 4. Installation

## 4.1 Base Docker image
Download the sample docker file from the docker hub.
You need the version that runs CUDA9.

## 4.2 Cuda9 and cuDNN7 libraries
Get the appropriate version of related CUDA libraries

```bash
dpkg --list | grep libnccl
dpkg --list | grep libcudnn
```
## 4.3 Anaconda Cloud and magma-cuda90

## 4.4 Prepare the Ubuntu EC2 Machine

### 4.4.1 Install Docker

### 4.4.2 Install nvidia-docker and nvidia-docker plugins

## 4.5 Dockerfile: Prepare PyTorch + Anaconda3 Docker Image


### 4.5.1 Cuda libraries

### 4.5.2 Anaconda3

### 4.5.3 magma-cuda90

### 4.5.4 Compile PyTorch for Volta

### 4.5.5 PyTorch Examples and Working Directory

## 4.6 Execution

### Build

```bash
nvidia-docker build -t danulab/pytorch-dlimg:17.10 -t danulab/pytorch-
dlimg:v1.0 -t danulab/pytorch-dlimg:latest .
```

Last lines:

```bash
Successfully tagged danulab/pytorch-dlimg:17.10
Successfully tagged danulab/pytorch-dlimg:v1.0
Successfully tagged danulab/pytorch-dlimg:latest
```
# 5. Test 

# 6. Conclusion




