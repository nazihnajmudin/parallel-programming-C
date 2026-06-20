#include "header/cuda.h"

#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

/* CUDA OPTIMIZER */
void get_optimal_config(int width, int height, dim3 *dim_grid, dim3 *dim_block) {
    
    // Dapatkan properti perangkat GPU 0
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    // Batasan perangkat keras
    const int max_threads_per_block = prop.maxThreadsPerBlock;
    const int warp_size = prop.warpSize;

    // Strategi optimasi:
    // Pilih ukuran blok yang merupakan kelipatan warp_size dan mendekati 256 atau 512
    // karena memberikan occupancy dan koalesensi yang baik
    int threads_per_block = 0;
    if (max_threads_per_block >= 512) {
        threads_per_block = 512;
    } else if (max_threads_per_block >= 256) {
        threads_per_block = 256;
    } else {
        threads_per_block = max_threads_per_block;
    }

    // Sesuaikan agar menjadi kelipatan warp_size
    threads_per_block = (threads_per_block / warp_size) * warp_size;
    if (threads_per_block == 0) { // Pastikan setidaknya sebesar warp_size
        threads_per_block = warp_size;
    }

    // Tentukan dimensi blok (persegi)
    int block_side = (int)sqrt((double)threads_per_block);
    
    // Sesuaikan block side agar menjadi kelipatan dari warp_size untuk dimensi x
    int dim_block_X = (block_side / warp_size) * warp_size;
    if (dim_block_X == 0) {
        dim_block_X = warp_size;
    }
    
    // Sesuaikan dim_block_Y agar total threads tetap optimal
    int dim_block_Y = threads_per_block / dim_block_X;
    
    // Fallback jika perhitungan tidak sesuai
    if (dim_block_X * dim_block_Y > max_threads_per_block) {
        dim_block_X = 16;
        dim_block_Y = 16;
    }

    // Menginisialisasi struct dim3 secara manual di C
    dim_block->x = dim_block_X;
    dim_block->y = dim_block_Y;

    // Tentukan dim_grid
    dim_grid->x = (width + dim_block->x - 1) / dim_block->x;
    dim_grid->y = (height + dim_block->y - 1) / dim_block->y;

    // Debug menggunakan printf
    printf("[Optimal CUDA Config]\n");
    printf("    GPU Model             : %s\n", prop.name);
    printf("    Warp size             : %d\n", warp_size);
    printf("    Threads/Block (target): %d\n", threads_per_block);
    printf("    Grid Dimension        : (%d, %d)\n", dim_grid->x, dim_grid->y);
    printf("    Block Dimension       : (%d, %d)\n", dim_block->x, dim_block->y);
}

