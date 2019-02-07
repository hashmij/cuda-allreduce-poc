#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/uio.h>
#include <sys/types.h>
#include <assert.h>
#include <cuda_runtime.h>

#include "allreduce_kernel.h"

#define _DEBUG_       0
#define SKIP_ITER     10
#define NUM_ITER      100

#define MIN_MSG_SIZE    1<<10
#define MAX_MSG_SIZE    1<<28

#define cudaCheckError() {                                          \
 cudaError_t e=cudaGetLastError();                                 \
 if(e!=cudaSuccess) {                                              \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
   exit(0); \
 }                                                                 \
}

extern void do_print_gpu(TYPE *buf, size_t nelems);
extern void do_memset_device_mt(TYPE *buf, size_t nelems, int v);
extern void do_allreduce_p2p(TYPE *sbuf, TYPE *rbuf, TYPE **peer_sbufs, TYPE **peer_rbuf, size_t count, int op, int rank, int size);


/* Utility functions at host side */
void print_buf_host(TYPE *buf, size_t nelems) {
    int i;
    for (i = 0; i < nelems; i++) {
        fprintf(stderr, "%d\n", buf[i]);
    }
}

void validate_buffer_host(TYPE *buf, size_t nelems, int v, int rank, int iter) {
    int i, err=0;

    for (i = 0; i < nelems; i++) {
        if (buf[i] != v) err++;
    }
    
    if (err > 0) {
        printf ("[%d-%d]: %d errors found ....\n", rank, iter, err);
    } else {
        printf ("[%d-%d]: no errors found ...\n", rank, iter);
    }
}

 

/* nvidia: p2pBandwidthTest */
int check_and_enable_peer_access(int me, int num_gpus)
{
    int peer;
    for (peer = 0; peer < num_gpus; peer++) {
    	int access;
    	if (peer != me) {
	        cudaDeviceCanAccessPeer(&access, me, peer);
	        if (!access) {
	    	    fprintf (stdout, "Device=%d %s Access Peer Device=%d\n", me, "CANNOT", peer);
		        return 1;
	        }
            cudaDeviceEnablePeerAccess(peer, 0);
	        cudaCheckError();
            cudaSetDevice(peer);
            cudaDeviceEnablePeerAccess(me, 0);
	        cudaCheckError();          
            cudaSetDevice(me);
    	}
    }
    return 0;
}

// ######## //

int main(int argc, char **argv)
{
    int i, j;
    int src = 0, skip = SKIP_ITER, iters = NUM_ITER;
    int rank, size;
    TYPE *d_sbuf, *d_rbuf;
    TYPE **d_peer_sbufs, **d_peer_rbufs;
    cudaIpcMemHandle_t l_sh, l_rh;
    cudaIpcMemHandle_t *d_peer_sh, *d_peer_rh;
    size_t bufsize = MAX_MSG_SIZE;
    size_t typesize = sizeof(TYPE);
    size_t nelems = bufsize / typesize;
    double start, end, duration;
    double min, max, avg, bw;
    int num_gpus = -1;
    int peer_access = 0;
    int op;
   
#if _DEBUG_
    void *h_validation_buf;
#endif
 
    /* MPI init */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* count devices */
    cudaGetDeviceCount(&num_gpus);
    cudaCheckError();

    /* each rank gets one device */  
    cudaSetDevice(rank);
    
    /* check for all-to-all connectivity */
    if ( (peer_access = check_and_enable_peer_access(rank, num_gpus)) > 0) { 
        
#if _DEBUG_
        fprintf (stdout, "Rank %d does not have all-to-all connectivity...\n", rank);
#endif
        goto cleanup;
    } else {
#if _DEBUG_
        fprintf (stdout, "Rank %d has all-to-all connectivity...enabled\n", rank);
#endif
    }
    
    /* create local sbuf and rbuf */
    cudaMalloc((void **)&d_sbuf, typesize * nelems);
    cudaMalloc((void **)&d_rbuf, typesize * nelems);
    cudaCheckError();

#if _DEBUG_
    printf ("debug\n");
    h_validation_buf = malloc(bufsize);
#endif
    
    /* create IPC handles of local device buffers */ 
    cudaIpcGetMemHandle(&l_sh, (void*)d_sbuf);
    cudaIpcGetMemHandle(&l_rh, (void*)d_rbuf);
    MPI_Barrier(MPI_COMM_WORLD);

    /* create remote device memory pointers */
    cudaMallocHost((void **)&d_peer_sbufs, size * sizeof (TYPE *)); 
    cudaMallocHost((void **)&d_peer_rbufs, size * sizeof (TYPE *));  
 
    /* create handles for remote memory */
    d_peer_sh = (cudaIpcMemHandle_t *) malloc(size * sizeof(cudaIpcMemHandle_t));
    d_peer_rh = (cudaIpcMemHandle_t *) malloc(size * sizeof(cudaIpcMemHandle_t));

    MPI_Barrier(MPI_COMM_WORLD);
    
    /* exchange handles */
    MPI_Allgather(&l_sh, sizeof(l_sh), MPI_BYTE, d_peer_sh, sizeof(l_sh), MPI_BYTE, MPI_COMM_WORLD);
    MPI_Allgather(&l_rh, sizeof(l_rh), MPI_BYTE, d_peer_rh, sizeof(l_rh), MPI_BYTE, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
 
    /* open handles with all other peers  */
    for (i = 0; i < size; i++) {
     if (i != rank) {
      cudaIpcOpenMemHandle((void **)&d_peer_sbufs[i], d_peer_sh[i], cudaIpcMemLazyEnablePeerAccess);
      cudaCheckError();
      cudaIpcOpenMemHandle((void **)&d_peer_rbufs[i], d_peer_rh[i], cudaIpcMemLazyEnablePeerAccess);
      cudaCheckError();
     } else {
       d_peer_sbufs[rank] = d_sbuf;
       d_peer_rbufs[rank] = d_rbuf;
     }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    
    // define operation to be used 
    op = SUM;
    
#if _DEBUG_
    printf ("[%d] host:: op %d, nelems %lu\n", rank, op, nelems);
#endif
   
    if (rank == 0)  {
        fprintf(stdout, "%10s %10s %13s %13s\n",                            \
            "NUM_GPUS", "MSG_SIZE (B)", "Latency (us)", "Bandwidth (MBps)");    \
    }


    size_t csize, cnelems;

    for (csize = MIN_MSG_SIZE; csize <= MAX_MSG_SIZE; csize<<=1) {
        
        cnelems = csize / typesize;
        // initialize the buffers by calling initializer kernel
        do_memset_device_mt(d_sbuf, cnelems, rank+1);
        do_memset_device_mt(d_rbuf, cnelems, 0);
     
        MPI_Barrier(MPI_COMM_WORLD);
        
        // Benchmark 
        start = MPI_Wtime();
        for (i=0; i<iters+skip; i++) {
            if (i==skip) start = MPI_Wtime();
#if _DEBUG_
            // Initialize the buffers by calling initializer kernel 
            do_memset_device_mt(d_sbuf, cnelems, rank+1);
            do_memset_device_mt(d_rbuf, cnelems, 0);
            MPI_Barrier(MPI_COMM_WORLD);
#endif
            // Actuall Allreduce kernel  
            do_allreduce_p2p(d_sbuf, d_rbuf, d_peer_sbufs, d_peer_rbufs, cnelems, op, rank, size);
#if _DEBUG_
            // validate the buffer against expected output
            int exp=0, j;
            for (j = 0; j < size; j++) { 
                exp = exp + (j + 1);
            }
            cudaMemcpy(h_validation_buf, d_rbuf, bufsize, cudaMemcpyDeviceToHost); 
            validate_buffer_host(h_validation_buf, cnelems, exp, rank, i);
#endif    
        }
        end = MPI_Wtime();
        duration = (end - start)*1e6/iters;     

        MPI_Barrier (MPI_COMM_WORLD);

        MPI_Reduce(&duration, &min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&duration, &max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&duration, &avg, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        avg = avg/size;
        bw = (1.0e6 * bufsize) / (avg * 1024 * 1024);

        if(rank == 0) {
            fprintf(stdout, "%8d %13lu %13.2lf %13.2lf\n", size, csize, avg, bw);
        }

        MPI_Barrier (MPI_COMM_WORLD);
    }

cleanup:
   
    MPI_Barrier (MPI_COMM_WORLD);
    
    if (peer_access > 0) {
        for (i = 0; i < size; i++) {
            cudaDeviceDisablePeerAccess(i);
            cudaCheckError();   
            /* close IPC handles */
            cudaIpcCloseMemHandle(&d_peer_sh[i]); 
            cudaIpcCloseMemHandle(&d_peer_rh[i]);            
        }
    } 
    
    MPI_Barrier (MPI_COMM_WORLD);
    cudaFree(d_sbuf);
    cudaFree(d_rbuf);
    cudaFree(d_peer_sbufs);
    cudaFree(d_peer_rbufs);
    free(d_peer_sh);
    free(d_peer_rh);
   
#if _DEBUG_
    free(h_validation_buf);
#endif 
    
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Finalize();
    return 0;
}
