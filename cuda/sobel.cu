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
    // Prepare for Image
    image_t *d_img;
    size_t size_p = h_img->w * h_img->h * sizeof(unsigned char);
    d_img->w = h_img->w;
    d_img->h = h_img->h;
    d_img->p = (unsigned char*)malloc(size_p);

    // Prepare other pointer
    int *d_mode;
    int *d_arr_thresholds;

    // GPU Memory Allocation
    cudaMalloc((void**)&d_img, sizeof(image_t));
    cudaMalloc((void**)&d_img->p, size_p);
    cudaMalloc((void**)&d_mode, sizeof(int));
    cudaMalloc((void**)&d_arr_thresholds, (*d_mode)*sizeof(int));
    
    // Copy to Device
    cudaMemcpy(d_img, &h_img, sizeof(image_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_img->p, &h_img->p, size_p, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mode, &h_mode, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_arr_thresholds, &h_arr_thresholds, *d_mode, cudaMemcpyHostToDevice);

    // Find best config
    dim3 dim_grid, dim_block;
    getOptimalConfig(h_img->w, h_img->h, &dim_grid, &dim_block);

    // Parallelize
    kernel_sobel<<<dim_grid, dim_block>>>(d_img, d_mode, d_arr_thresholds);

    // Copy to Host
    cudaMemcpy(h_img, d_img, sizeof(image_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_img->p, d_img->p, size_p, cudaMemcpyDeviceToHost);

    // GPU Memory Deallocation
    cudaFree(d_arr_thresholds);
    cudaFree(d_mode);
    cudaFree(d_img->p);
    cudaFree(d_img);

    // CPU Memory Deallocation
    free(d_img->p);
}