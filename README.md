

MVAPICH2-X with basic CUDA and XPMEM support:
=============================================

 $ ./configure --enable-xpmem --with-xpmem=/opt/xpmem \
    --with-xpmem-include=/opt/xpmem/include --with-xpmem-libpath=/opt/xpmem/lib \
    --enable-cuda --with-cuda-include=/opt/cuda/9.2/include \
    --with-cuda-libpath=/opt/cuda/9.2/lib64 CFLAGS="-I/opt/xpmem/include \
    -I/opt/cuda/9.2/include" LDFLAGS="-Wl,-rpath,/opt/xpmem/lib -L/opt/xpmem/lib \
    -lxpmem -Wl,-rpath,/opt/cuda/9.2/lib64 -L/opt/cuda/9.2/lib64 -lcuda" \
    --prefix=`pwd`/install

 $ make -j 12 && make install
 

Running
=======
mpirun_rsh -np 4 -hostfile ~/hosts MV2_SMP_USE_CMA=0 MV2_USE_CUDA=1 get_local_rank ./allreduce_p2p.x D D