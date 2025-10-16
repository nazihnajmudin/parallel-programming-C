#ifndef CUDA_H
#define CUDA_H

#include "image.h"
#include <cstdio>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));     \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

/* ALGORITMA SOBEL */
void sobel(image_t *h_img, int h_mode, int *h_arr_thresholds);

/* CUDA OPTIMIZER */
void get_optimal_config(int width, int height, dim3 *grid_dim, dim3 *dim_block);

#endif