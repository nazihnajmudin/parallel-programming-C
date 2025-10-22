#ifndef CUDA_H
#define CUDA_H

#include "image.h"
#include <cstdio>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));     \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

struct cudaExecTime_t {
    float kernel_time;
    float malloc_time;
    float memcpy_time;
    float cufree_time;
    float cuda_events;
};
float total_cuda_time(cudaExecTime_t *c);

/* ALGORITMA SOBEL */
cudaExecTime_t sobel(image_t *h_img, int h_mode, int *h_arr_thresholds, dim3 *grid_dim, dim3 *dim_block, unsigned char cuda_type);
__global__ void kernel_sobel_raw(unsigned char *in, unsigned char *out, int w, int h, int mode, int *d_thresholds);
__global__ void kernel_sobel_shared(unsigned char *in, unsigned char *out, int w, int h, int mode, int *d_thresholds);
__global__ void kernel_sobel_tiled(const unsigned char* __restrict__ in, unsigned char* out,int w, int h, int mode, const int* d_thresholds);

/* CUDA OPTIMIZER */
void get_optimal_config(int width, int height, dim3 *grid_dim, dim3 *dim_block);

#endif