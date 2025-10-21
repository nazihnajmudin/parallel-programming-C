#include "header/cuda.h"
#include "header/image.h"

cudaExecTime_t sobel(image_t *h_img, int mode, int *h_arr_thresholds, dim3 *dim_grid, dim3 *dim_block, unsigned char cuda_type) {
    
    cudaExecTime_t result;
    result.kernel_time = 0;
    result.malloc_time = 0;
    result.memcpy_time = 0;
    result.cufree_time = 0;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    unsigned char *d_p, *d_p_out;
    int *d_arr_thresholds;

    size_t size_p = h_img->w * h_img->h * sizeof(unsigned char);
    size_t smem_size = (dim_block->x + 2) * (dim_block->y + 2) * sizeof(unsigned char);
    
    
    // GPU Memory Allocation
    cudaEventRecord(start);
    CUDA_CHECK(cudaMalloc((void**)&d_p, size_p));
    CUDA_CHECK(cudaMalloc((void**)&d_p_out, size_p));
    CUDA_CHECK(cudaMalloc((void**)&d_arr_thresholds, max(mode, 1) * sizeof(int)));
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&result.malloc_time, start, stop);
    
    
    // Copy image data and params
    cudaEventRecord(start);
    unsigned char *h_pinned;
    
    CUDA_CHECK(cudaMallocHost((void**)&h_pinned, size_p));
    memcpy(h_pinned, h_img->p, size_p);
    CUDA_CHECK(cudaMemcpyAsync(d_p, h_pinned, size_p, cudaMemcpyHostToDevice));
    
    CUDA_CHECK(cudaMemcpy(d_arr_thresholds, h_arr_thresholds, mode * sizeof(int), cudaMemcpyHostToDevice));
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&result.memcpy_time, start, stop);
    
    
    dim3 grid(  (h_img->w + BLOCK_SIZE - 1) / BLOCK_SIZE,
    (h_img->h + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    
    // Kernel launch
    cudaEventRecord(start);
    if (cuda_type =='r') {          // cuda_raw
        kernel_sobel_raw<<<*dim_grid, *dim_block>>>(d_p, d_p_out, h_img->w, h_img->h, mode, d_arr_thresholds);
    } else if (cuda_type == 's') {  // cuda_raw 16 x 16
        kernel_sobel_raw<<<grid, block>>>(d_p, d_p_out, h_img->w, h_img->h, mode, d_arr_thresholds);
    } else if (cuda_type == 'm') {  // cuda_shared memory
        kernel_sobel_shared<<<grid, block>>>(d_p, d_p_out, h_img->w, h_img->h, mode, d_arr_thresholds);
    } else if (cuda_type == 'd') {  // cuda_shared_dyn
        kernel_sobel_tiled<<<*dim_grid, *dim_block, smem_size>>>(d_p, d_p_out, h_img->w, h_img->h, mode, d_arr_thresholds);
    }
    // CUDA_CHECK(cudaGetLastError());         // Check launch errors
    // CUDA_CHECK(cudaDeviceSynchronize());    // Check execution errors <-- katanya ini memperlambat
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&result.kernel_time, start, stop);
    

    // Copy back results
    cudaEventRecord(start);
    CUDA_CHECK(cudaMemcpy(h_pinned, d_p_out, size_p, cudaMemcpyDeviceToHost));
    memcpy(h_img->p, h_pinned, size_p);
    CUDA_CHECK(cudaFreeHost(h_pinned));
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float memcpy_time_temp = result.memcpy_time;
    cudaEventElapsedTime(&memcpy_time_temp, start, stop);
    result.memcpy_time += memcpy_time_temp;
    
    
    // Free device memory
    cudaEventRecord(start);
    CUDA_CHECK(cudaFree(d_arr_thresholds));
    CUDA_CHECK(cudaFree(d_p));
    CUDA_CHECK(cudaFree(d_p_out));
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&result.cufree_time, start, stop);
    

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return result;
}