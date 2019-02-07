#include <stdio.h>
#include <cuda.h>
#include "allreduce_kernel.h"

/* =============== Reduction Phase Kernels =============== */
__global__ void allreduce_kernel(TYPE *sbuf, TYPE *rbuf, TYPE **peer_sbufs, TYPE **peer_rbufs, 
                                    size_t count, int op, int lrank, int lsize) {

	TYPE *dst_buf, *peer_sbuf, *peer_rbuf;
	int chunk = (count % lsize == 0) ? count / lsize : count / lsize + count % lsize; 
	int offset = lrank * chunk;
	int end = offset + chunk;

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
    int p, peer;

 //   printf ("[%d]: tid %d, stride %d, offset %d, end %d\n", lrank, tid, stride, offset, end);
	
	dst_buf = rbuf;
	
	if (op == SUM) {
		// everyone copy your own chunk into your dst_buf
		for (int i = offset+tid; i < end; i+=stride) {
            dst_buf [i] = sbuf[i];
            
            // This synchronization is *very* important as it synchronizes all the 
            // threads in thread-block to wait before they proceed to the next step.
            // As this step ensures the first copy of local recvbuf into local sbuf 
            // and ensures that there is no garbage data before the *op* is applied 
            // in the next phase.
            __syncthreads();
		} 

        /* perform reduce operations */
		for (p = 1; p < lsize; p++) {
			// Perform operation on peer's buffer and save resul in destination buffer 
			peer = (lrank + p) % lsize;
            peer_sbuf = peer_sbufs[peer];
			for (int i = offset+tid; i < end; i+=stride) {
                dst_buf[i] += peer_sbuf[i];
                __syncthreads();
			} 
		}

		// Perform broadcast of Reduced data
		for (p = 1; p < lsize; p++) {
			peer = (lrank + p) % lsize;
            peer_rbuf = peer_rbufs[peer];
			for (int i = offset+tid; i < end; i+=stride) {
                peer_rbuf[i] = dst_buf [i];
                __syncthreads();
			}
		}
  
    } else if (op == MUL) {
        printf ("op is %d and nelems %lu\n", op, count);
    }
    
}
/* ==================== Utility Kernels ================== */

__global__ void print_kernel(TYPE *buf, size_t nelems) {
    for (int i = 0; i < nelems; i++) {
        printf("%d ", buf[i]);
    }
    printf("\n");
    __syncthreads();
}

__global__ void memset_kernel_mt(TYPE *buf, size_t nelems, int v) {
    /* multiple threads loop over the data and set the values */
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x; 
    int i;
    for (i = tid; i < nelems; i+=stride) {
        buf[i] = v;
    }
}

__global__ void memset_kernel(TYPE *buf, size_t nelems, int v) {
    for (int i = 0; i < nelems; i++) {
        buf[i] = v;
    }
}

/* ==================== Externs for kernels ================== */
extern "C" void do_allreduce_p2p(TYPE *sbuf, TYPE *rbuf, TYPE **peer_sbufs, TYPE **peer_rbufs, size_t count, int op, int rank, int size) {
    //allreduce_kernel<<<32, 1024>>>(sbuf, rbuf, peer_sbufs, peer_rbufs, count, op, rank, size);
    allreduce_kernel<<<64, 1024>>>(sbuf, rbuf, peer_sbufs, peer_rbufs, count, op, rank, size);
    cudaDeviceSynchronize();
}

extern "C" void do_memset_device_mt(TYPE *buf, size_t nelems, int v) {
    memset_kernel_mt<<< 32, 256>>>(buf, nelems, v);
    cudaDeviceSynchronize();
}

extern "C" void do_memset_device(TYPE *buf, size_t nelems, int v) {
    memset_kernel<<< 1, 1 >>>(buf, nelems, v);
    cudaDeviceSynchronize();
}

extern "C" void do_print_gpu(TYPE *buf, size_t nelems) {
    print_kernel<<< 1, 1 >>>(buf, nelems);
    cudaDeviceSynchronize();
}

