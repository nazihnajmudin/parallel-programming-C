#ifndef CUDA_HPP
#define CUDA_HPP

#include "image.h"
#include <cuda_runtime.h>

/* ALGORITMA SOBEL */
void sobel(image_t *h_img, int h_mode, int *h_arr_thresholds);

/* CUDA OPTIMIZER */
void getOptimalConfig(int width, int height, dim3 *grid_dim, dim3 *dim_block);

#endif