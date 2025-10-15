#include "header/cuda.h"
#include "header/image.h"
#include <cstdio>

__device__ int count_convolution(image_t *img, int x, int y) {
    if ((x > 0 && x < img->w-1) && (y > 0 && y < img->h-1)) {
        int ver = (pixel(img, x-1, y-1) + pixel(img, x+1, y-1))
                - (pixel(img, x-1, y+1) + pixel(img, x+1, y+1))
                + (2 * pixel(img, x, y-1)) - (2 * pixel(img, x, y+1));
        int hor = (pixel(img, x+1, y-1) + pixel(img, x+1, y+1))
                - (pixel(img, x-1, y-1) + pixel(img, x-1, y+1))
                + (2 * pixel(img, x+1, y)) - (2 * pixel(img, x-1, y));
        return (int)sqrt(pow(ver, 2) + pow(hor, 2));
    }
    return 0;
}

__global__ void kernel_sobel(image_t *d_img, int *d_mode, int *d_thresholds) {
    // Cara indexing gambar sudah sesuai dengan indexing di cuda
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    // Count G
    int g = count_convolution(d_img, x, y);
    __syncthreads();
    // Output
    if (*d_mode == 0) {
        pixel(d_img, x, y) = min(g, 255);
    } else if (*d_mode == 1) {
        if (g > d_thresholds[0]) pixel(d_img, x, y) = 0;
        else pixel(d_img, x, y) = 255;
    } else {
        // d_mode = k = bins-1
        int bins = *d_mode + 1;
        int idx = 0;
        while (idx < *d_mode && g > d_thresholds[idx]) idx++;
        pixel(d_img, x, y) = (255 * idx) / (bins - 1);
    }
}

void sobel(image_t *h_img, int h_mode, int *h_arr_thresholds) {
    image_t *d_img;
    unsigned char *d_p;
    int *d_mode;
    int *d_arr_thresholds;

    size_t size_p = h_img->w * h_img->h * sizeof(unsigned char);

    // GPU Memory Allocation
    CUDA_CHECK(cudaMalloc((void**)&d_img, sizeof(image_t)));
    CUDA_CHECK(cudaMalloc((void**)&d_p, size_p));
    CUDA_CHECK(cudaMalloc((void**)&d_mode, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_arr_thresholds, h_mode * sizeof(int)));

    // Copy image data
    CUDA_CHECK(cudaMemcpy(d_p, h_img->p, size_p, cudaMemcpyHostToDevice));

    // Prepare image_t struct on host
    image_t temp_img = *h_img;
    temp_img.p = d_p;

    // Copy struct & other parameters
    CUDA_CHECK(cudaMemcpy(d_img, &temp_img, sizeof(image_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mode, &h_mode, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_arr_thresholds, h_arr_thresholds, h_mode * sizeof(int), cudaMemcpyHostToDevice));

    // Find best config
    dim3 dim_grid, dim_block;
    getOptimalConfig(h_img->w, h_img->h, &dim_grid, &dim_block);

    // Kernel launch
    kernel_sobel<<<dim_grid, dim_block>>>(d_img, d_mode, d_arr_thresholds);
    CUDA_CHECK(cudaGetLastError());         // Check launch errors
    CUDA_CHECK(cudaDeviceSynchronize());    // Check execution errors

    // Copy back results
    CUDA_CHECK(cudaMemcpy(h_img->p, d_p, size_p, cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK(cudaFree(d_arr_thresholds));
    CUDA_CHECK(cudaFree(d_mode));
    CUDA_CHECK(cudaFree(d_p));
    CUDA_CHECK(cudaFree(d_img));
}