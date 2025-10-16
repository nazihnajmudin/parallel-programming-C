#include "header/cuda.h"

#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

/* CUDA OPTIMIZER */
void getOptimalConfig(int width, int height, dim3 *dim_grid, dim3 *dim_block) {
    
    // Dapatkan properti perangkat GPU 0
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    // Batasan perangkat keras
    const int maxThreadsPerBlock = prop.maxThreadsPerBlock;
    const int warpSize = prop.warpSize;

    // Strategi optimasi:
    // Pilih ukuran blok yang merupakan kelipatan warpSize dan mendekati 256 atau 512
    // karena memberikan occupancy dan koalesensi yang baik
    int threadsPerBlock = 0;
    if (maxThreadsPerBlock >= 512) {
        threadsPerBlock = 512;
    } else if (maxThreadsPerBlock >= 256) {
        threadsPerBlock = 256;
    } else {
        threadsPerBlock = maxThreadsPerBlock;
    }

    // Sesuaikan agar menjadi kelipatan warpSize
    threadsPerBlock = (threadsPerBlock / warpSize) * warpSize;
    if (threadsPerBlock == 0) { // Pastikan setidaknya sebesar warpSize
        threadsPerBlock = warpSize;
    }

    // Tentukan dimensi blok (persegi)
    int blockSide = (int)sqrt((double)threadsPerBlock);
    
    // Sesuaikan blockSide agar menjadi kelipatan dari warpSize untuk dimensi x
    int dim_block_X = (blockSide / warpSize) * warpSize;
    if (dim_block_X == 0) {
        dim_block_X = warpSize;
    }
    
    // Sesuaikan dim_block_Y agar total threads tetap optimal
    int dim_block_Y = threadsPerBlock / dim_block_X;
    
    // Fallback jika perhitungan tidak sesuai
    if (dim_block_X * dim_block_Y > maxThreadsPerBlock) {
        dim_block_X = 16;
        dim_block_Y = 16;
    }

    // Menginisialisasi struct dim3 secara manual di C
    dim_block->x = dim_block_X;
    dim_block->y = dim_block_Y;
    // dim_block->z = 1;

    // Tentukan dim_grid
    dim_grid->x = (width + dim_block->x - 1) / dim_block->x;
    dim_grid->y = (height + dim_block->y - 1) / dim_block->y;
    // dim_grid->z = 1;

    // Debug menggunakan printf
    printf("[Optimal CUDA Config]\n");
    printf("  GPU: %s\n", prop.name);
    printf("  Warp size: %d\n", warpSize);
    printf("  Threads/Block (target): %d\n", threadsPerBlock);
    printf("  dim_block: (%d, %d)\n", dim_block->x, dim_block->y);
    printf("  dim_grid: (%d, %d)\n", dim_grid->x, dim_grid->y);
}

