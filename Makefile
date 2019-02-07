# Paths
P_DIR=$(PWD)
SRC_DIR=$(PWD)/src
OBJ_DIR=$(PWD)/obj
EXE_DIR=$(PWD)/bin

# Dependecy libraries
MPI_INSTALL=/home/hashmi/gpu-work/mvapich2/install
CUDA_INSTALL=/opt/packages/cuda/9.2

# Compiler
NVCC = $(CUDA_INSTALL)/bin/nvcc
MPICC = $(MPI_INSTALL)/bin/mpicc

CFLAGS = -g -I$(MPI_INSTALL)/include -I$(CUDA_INSTALL)/include
LDFLAGS = -L$(CUDA_INSTALL)/lib64

OBJS = allreduce_kernel.o allreduce_p2p.o

all: $(OBJS)
	$(MPICC) $(CFLAGS) $(LDFLAGS) $(OBJS) -o allreduce_p2p.x -lcudart -lstdc++ 

allreduce_kernel.o: $(SRC_DIR)/allreduce_kernel.cu
	$(NVCC) -c $(SRC_DIR)/allreduce_kernel.cu
	
allreduce_p2p.o: $(SRC_DIR)/allreduce_p2p.c 
	$(MPICC) -c $(SRC_DIR)/allreduce_p2p.c $(LDFLAGS) -lcudart

clean:
	rm -f *.o *.x
