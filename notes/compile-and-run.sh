nvcc -c um_kernel.cu -o um_kernel.o
mpicc -c -L/opt/cuda/9.2/lib64 xpmem_get_cached_um.c -lcudart
mpicc xpmem_get_cached_um.o um_kernel.o -lcudart -L/opt/cuda/9.2/lib64 -o exe

mpirun_rsh -np 2 -hostfile ~/hosts MV2_CPU_BINDING_POLICY=scatter MV2_SHOW_CPU_BINDING=1 MV2_DEBUG_CORESIZE=unlimited exe
