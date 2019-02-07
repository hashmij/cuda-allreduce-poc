#/bin/sh

module load cuda/9.2

MPI=/home/hashmi/gpu-work/mvapich2/install
CUDA_HOME=/opt/packages/cuda/9.2

export PATH=$MPI/bin:$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$MPI/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH
