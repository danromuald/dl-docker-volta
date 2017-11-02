#!/bin/bash
set -e
cat <<EOF

#############################################################################################
#                          This Docker container comes with:                                #
#                        PyTorch + Full Anaconda3 + Python3.5                               #
# CUDA SDK 9.0 supports compute capability 3.0 through 7.x (Kepler, Maxwell, Pascal, Volta) #
#                                                                                           #
#  PyTorch Built for:                                                                       #
#    - Kepler: 3.7 (AWS P2) +PTX                                                            #
#    - Maxwell: 5.0 5.2                                                                     #
#    - Jetson TX1: 5.3                                                                      #
#    - Pascal P100: 6.0                                                                     #
#    - Pascal GTX family: 6.1                                                               #
#    - Jetson TX2: 6.2                                                                      #
#    - Volta V100: 7.0+PTX (PTX = parallel Thread Execution): even faster!                  #
#                                                                                           #
   Release ${DANULAB_PYTORCH_IMAGE_VERSION} (build ${DANULAB_PYTORCH_BUILD_VERSION})       
#                                                                                           #
#                                                                                           #
#                                                                                           #
#############################################################################################

EOF

if [[ "$(find /usr -name libcuda.so.1) " == " " || "$(ls /dev/nvidiactl) " == " " ]]; then
  echo
  echo "WARNING: The NVIDIA Driver was not detected.  GPU functionality will not be available."
  echo "   Use 'nvidia-docker run' to start this container; see"
  echo "   https://github.com/NVIDIA/nvidia-docker/wiki/nvidia-docker ."
fi

if [[ "$(df -k /dev/shm |grep ^shm |awk '{print $2}') " == "65536 " ]]; then
  echo
  echo "NOTE: The SHMEM allocation limit is set to the default of 64MB.  This may be"
  echo "   insufficient for PyTorch.  NVIDIA recommends the use of the following flags:"
  echo "   nvidia-docker run --ipc=host ..."
fi

echo

nvidia-smi

cd ${PYTORCH_WORK_DIR}

if [[ $# -eq 0 ]]; then
  exec "/bin/bash"
else
  exec "$@"
fi