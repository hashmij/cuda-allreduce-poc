#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <assert.h>

#define MAX_BLOCKS 2048

#define cudaCheckError() {                                          \
 cudaError_t e=cudaGetLastError();                                 \
 if(e!=cudaSuccess) {                                              \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
   exit(0); \
 }                                                                 \
}

double getMicrosecondTimeStamp()
{
    double retval;
    struct timeval tv;
    if (gettimeofday(&tv, NULL)) {
        perror("gettimeofday");
        abort();
    }
    retval = ((double)tv.tv_sec) * 1000000 + tv.tv_usec;
    return retval;
}
#define TIME() getMicrosecondTimeStamp()

__global__ void init_arrays_kernel(int* d_in, int* d_out, int N) { 
  int idx = blockIdx.x * blockDim.x + threadIdx.x; 
  for (int i = idx; i < N; i += blockDim.x * gridDim.x) { 
    d_out[i] = 0;
    d_in[i] = 1; 
  } 
} 

void init_arrays(int* d_in, int* d_out, int N) 
{ 
    int threads = 128; 
    int blocks = min((N + threads-1) / threads, MAX_BLOCKS);  
    init_arrays_kernel<<<blocks, threads>>>(d_in, d_out, N); 
    cudaDeviceSynchronize();
}

// ############################# SCALAR ######################### //
__global__ void device_copy_scalar_kernel(int* d_in, int* d_out, int N) { 
  int idx = blockIdx.x * blockDim.x + threadIdx.x; 
  for (int i = idx; i < N; i += blockDim.x * gridDim.x) { 
    d_out[i] = d_in[i]; 
  } 
} 

void device_copy_scalar(int* d_in, int* d_out, int N) 
{ 
    int threads = 128; 
    int blocks = min((N + threads-1) / threads, MAX_BLOCKS);  
    device_copy_scalar_kernel<<<blocks, threads>>>(d_in, d_out, N); 
    cudaDeviceSynchronize();
}

// ############################# VEC 2 ######################### //
__global__ void device_copy_vector2_kernel(int* d_in, int* d_out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < N/2; i += blockDim.x * gridDim.x) {
        reinterpret_cast<int2*>(d_out)[i] = reinterpret_cast<int2*>(d_in)[i];
    }

    // in only one thread, process final element (if there is one)
    if (idx==N/2 && N%2==1)
        d_out[N-1] = d_in[N-1];
}

void device_copy_vector2(int* d_in, int* d_out, int N) {
    int threads = 128; 
    int blocks = min((N/2 + threads-1) / threads, MAX_BLOCKS); 
    device_copy_vector2_kernel<<<blocks, threads>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
}


// ############################# VEC 4 ######################### //
__global__ void device_copy_vector4_kernel(int* d_in, int* d_out, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  for(int i = idx; i < N/4; i += blockDim.x * gridDim.x) {
      reinterpret_cast<int4*>(d_out)[i] = reinterpret_cast<int4*>(d_in)[i];
  }

  // in only one thread, process final elements (if there are any)
  int remainder = N%4;
  if (idx==N/4 && remainder!=0) {
    while(remainder) {
        int idx = N - remainder--;
        d_out[idx] = d_in[idx];
    }
  }
}

void device_copy_vector4(int* d_in, int* d_out, int N) {
    int threads = 128;
    int blocks = min((N/4 + threads-1) / threads, MAX_BLOCKS);

    device_copy_vector4_kernel<<<blocks, threads>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
}

    
// ############################################################# //

int main (int argc, char **argv)
{

    const long N = 1024 * 1024 * 64;
    int *d_src, *d_dst, *h_val;
    double t_start, scalar_lat, vec64_lat, vec128_lat;
    int iter = 1000;
    int skip = 100;

    cudaMalloc((void **)&d_src, N * sizeof(int));
    cudaCheckError();
    cudaMalloc((void **)&d_dst, N * sizeof(int));
    cudaCheckError();

    init_arrays(d_src, d_dst, N);
    cudaCheckError();


    fprintf(stdout, "%10s %10s %13s %13s\n",                            \
            "MSG_SIZE (B)", "Scalar Lat. (us)", "Vec64 Lat. (us)",  "Vec128 Lat. (us)");    \


     
    for (unsigned long size = 128; size <= 1<<28; size <<=1) {
    
        for (int i = 0; i < iter+skip; i++) {
            if (i == skip) t_start = TIME();
            device_copy_scalar(d_src, d_dst, size/sizeof(int));
        }
        scalar_lat = (TIME() - t_start)/iter;
        
        for (int i = 0; i < iter+skip; i++) {
            if (i == skip) t_start = TIME();
            device_copy_vector2(d_src, d_dst, size/sizeof(int));
        }
        vec64_lat = (TIME() - t_start)/iter;

        for (int i = 0; i < iter+skip; i++) {
            if (i == skip) t_start = TIME();
            device_copy_vector4(d_src, d_dst, size/sizeof(int));
        }
        vec128_lat = (TIME() - t_start)/iter;
    
        fprintf(stdout, "%13lu %13.2lf %13.2lf %13.2lf\n", size, scalar_lat, vec64_lat, vec128_lat);
    }


    h_val = (int *)malloc(N*sizeof(int));
    cudaMemcpy(h_val, d_dst, N*sizeof(int), cudaMemcpyDeviceToHost); 

    for (int i = 0; i < N; i++) {
        //printf ("%d \n", h_val[i]);
        assert(h_val[i] == 1);
    }
    cudaFree(d_src);
    cudaFree(d_dst);

    return 0;
} 
