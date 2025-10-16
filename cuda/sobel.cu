#include "header/cuda.h"
#include "header/image.h"

__global__ void kernel_sobel_raw(unsigned char *in, unsigned char *out, int w, int h, int mode, int *d_thresholds) {
    // Cara indexing gambar sudah sesuai dengan indexing di cuda
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    // Convolution
    int ver = 0, hor = 0;
    if (x > 0 && x < w-1 && y > 0 && y < h-1) {
        ver = in[(y+1)*w + (x-1)] + 2*in[(y+1)*w + x] + in[(y+1)*w + (x+1)]
        - (in[(y-1)*w + (x-1)] + 2*in[(y-1)*w + x] + in[(y-1)*w + (x+1)]);
        hor = in[(y-1)*w + (x+1)] + 2*in[y*w + (x+1)] + in[(y+1)*w + (x+1)]
        - (in[(y-1)*w + (x-1)] + 2*in[y*w + (x-1)] + in[(y+1)*w + (x-1)]);
    }
    int g = (int)sqrtf(ver*ver + hor*hor);
    
    // Output
    if (mode == 0) {
        out[y*w + x] = min(g, 255);
    } else if (mode == 1) {
        out[y*w + x] = (g > d_thresholds[0]) ? 0 : 255;
    } else {
        int bins = mode + 1;
        int idx = 0;
        while (idx < mode && g > d_thresholds[idx]) idx++;
        out[y*w + x] = (255 * idx) / (bins - 1);
    }
}

void sobel(image_t *h_img, int mode, int *h_arr_thresholds) {

    unsigned char *d_p, *d_p_out;
    int *d_arr_thresholds;

    size_t size_p = h_img->w * h_img->h * sizeof(unsigned char);

    // GPU Memory Allocation
    CUDA_CHECK(cudaMalloc((void**)&d_p, size_p));
    CUDA_CHECK(cudaMalloc((void**)&d_p_out, size_p));
    CUDA_CHECK(cudaMalloc((void**)&d_arr_thresholds, max(mode, 1) * sizeof(int)));

    // Copy image data
    CUDA_CHECK(cudaMemcpy(d_p, h_img->p, size_p, cudaMemcpyHostToDevice));

    // Copy struct & other parameters
    CUDA_CHECK(cudaMemcpy(d_arr_thresholds, h_arr_thresholds, mode * sizeof(int), cudaMemcpyHostToDevice));

    // Find best config
    dim3 dim_grid, dim_block;
    get_optimal_config(h_img->w, h_img->h, &dim_grid, &dim_block);

    // Kernel launch
    kernel_sobel_raw<<<dim_grid, dim_block>>>(d_p, d_p_out, h_img->w, h_img->h, mode, d_arr_thresholds);
    CUDA_CHECK(cudaGetLastError());         // Check launch errors
    CUDA_CHECK(cudaDeviceSynchronize());    // Check execution errors

    // Copy back results
    CUDA_CHECK(cudaMemcpy(h_img->p, d_p_out, size_p, cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK(cudaFree(d_arr_thresholds));
    CUDA_CHECK(cudaFree(d_p));
    CUDA_CHECK(cudaFree(d_p_out));
}