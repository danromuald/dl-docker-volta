# Building a PyTorch Volta Deep Learning container on AWS EC2 P3

# 1. What you get

- PyTorch
- Full Anaconda3 with Scipy, Nunpy, Matplotlit, Scikit-learn, mkl, Jupyter and more
- Nvidia Volta GPU
- Cuda9 SDK
- PyTorch examples


# 2. Environment and Prerequisites

- Amazon EC2 P3 instance
- Ubuntu image

# 3. Supported GPU Compute Capabilities


- Kepler: 3.7+PTX, for the AWS EC2 P2 instances.
- Maxwell: 5.0 5.2                                             
- Jetson TX1: 5.3                                              
- Pascal P100: 6.0                                             
- Pascal GTX family: 6.1                                       
- Jetson TX2: 6.2                                             
- Volta V100: 7.0+PTX           

```PTX = Parallel Thread Execution```.

# 4. Installation

# 4.1 Launch an Ubuntu16.04 Machine

## Todo:
    - Add img for the mainch

[prepare](../prepare_ubuntu-xenial-amd64.sh)

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

```Bash
nvidia-docker build -t danulab/pytorch-dlimg:17.10 -t danulab/pytorch-
dlimg:v1.0 -t danulab/pytorch-dlimg:latest .
```

Last lines:

```Bash
Successfully tagged danulab/pytorch-dlimg:17.10
Successfully tagged danulab/pytorch-dlimg:v1.0
Successfully tagged danulab/pytorch-dlimg:latest
```
# 5. Test 


# 6. Conclusion




