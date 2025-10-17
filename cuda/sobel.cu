#include "header/cuda.h"
#include "header/image.h"

void sobel(image_t *h_img, int mode, int *h_arr_thresholds, dim3 *dim_grid, dim3 *dim_block, unsigned char cuda_type) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    unsigned char *d_p, *d_p_out;
    int *d_arr_thresholds;

    size_t size_p = h_img->w * h_img->h * sizeof(unsigned char);

    // GPU Memory Allocation
    CUDA_CHECK(cudaMalloc((void**)&d_p, size_p));
    CUDA_CHECK(cudaMalloc((void**)&d_p_out, size_p));
    CUDA_CHECK(cudaMalloc((void**)&d_arr_thresholds, max(mode, 1) * sizeof(int)));

    // Copy image data
    unsigned char *h_pinned;
    CUDA_CHECK(cudaMallocHost((void**)&h_pinned, size_p));
    memcpy(h_pinned, h_img->p, size_p);
    CUDA_CHECK(cudaMemcpyAsync(d_p, h_pinned, size_p, cudaMemcpyHostToDevice));
    
    // Copy struct & other parameters
    CUDA_CHECK(cudaMemcpy(d_arr_thresholds, h_arr_thresholds, mode * sizeof(int), cudaMemcpyHostToDevice));
    
    // Kernel launch
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((h_img->w + BLOCK_SIZE - 1) / BLOCK_SIZE,
    (h_img->h + BLOCK_SIZE - 1) / BLOCK_SIZE);
    size_t smem_size = (dim_block->x + 2) * (dim_block->y + 2) * sizeof(unsigned char);
    switch (cuda_type) {
        case 'r': // cuda_raw
            kernel_sobel_raw<<<*dim_grid, *dim_block>>>(d_p, d_p_out, h_img->w, h_img->h, mode, d_arr_thresholds);
            break;
        case 's': // cuda_raw 16 x 16
            kernel_sobel_raw<<<grid, block>>>(d_p, d_p_out, h_img->w, h_img->h, mode, d_arr_thresholds);
            break;
        case 'm': // cuda_shared memory
            kernel_sobel_shared<<<grid, block>>>(d_p, d_p_out, h_img->w, h_img->h, mode, d_arr_thresholds);
            break;
        case 'd': // cuda_shared_dyn
            kernel_sobel_tiled<<<*dim_grid, *dim_block, smem_size>>>(d_p, d_p_out, h_img->w, h_img->h, mode, d_arr_thresholds);
            break;
        default:
            break;
    }
    // CUDA_CHECK(cudaGetLastError());         // Check launch errors
    // CUDA_CHECK(cudaDeviceSynchronize());    // Check execution errors <-- katanya ini memperlambat

    // Copy back results
    CUDA_CHECK(cudaMemcpy(h_pinned, d_p_out, size_p, cudaMemcpyDeviceToHost));
    memcpy(h_img->p, h_pinned, size_p);
    CUDA_CHECK(cudaFreeHost(h_pinned));

    // Free device memory
    CUDA_CHECK(cudaFree(d_arr_thresholds));
    CUDA_CHECK(cudaFree(d_p));
    CUDA_CHECK(cudaFree(d_p_out));

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("CUDA event benchmark: %f ms\n", milliseconds);
}