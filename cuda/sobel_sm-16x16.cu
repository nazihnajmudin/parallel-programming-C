#include "header/cuda.h"
#include "header/image.h"

__global__ void kernel_sobel_shared(unsigned char *in, unsigned char *out, int w, int h, int mode, int *d_thresholds) {
    __shared__ unsigned char tile[BLOCK_SIZE + 2][BLOCK_SIZE + 2];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * BLOCK_SIZE + tx;
    int y = blockIdx.y * BLOCK_SIZE + ty;

    // Global index
    int gx = min(max(x, 0), w - 1);
    int gy = min(max(y, 0), h - 1);

    // Load main pixel
    tile[ty + 1][tx + 1] = in[gy * w + gx];

    // Load halo region (edges of tile)
    if (tx == 0 && gx > 0)
        tile[ty + 1][0] = in[gy * w + gx - 1];
    if (tx == BLOCK_SIZE - 1 && gx < w - 1)
        tile[ty + 1][BLOCK_SIZE + 1] = in[gy * w + gx + 1];
    if (ty == 0 && gy > 0)
        tile[0][tx + 1] = in[(gy - 1) * w + gx];
    if (ty == BLOCK_SIZE - 1 && gy < h - 1)
        tile[BLOCK_SIZE + 1][tx + 1] = in[(gy + 1) * w + gx];

    // Load corners (optional, for correctness)
    if (tx == 0 && ty == 0 && gx > 0 && gy > 0)
        tile[0][0] = in[(gy - 1) * w + gx - 1];
    if (tx == BLOCK_SIZE - 1 && ty == 0 && gx < w - 1 && gy > 0)
        tile[0][BLOCK_SIZE + 1] = in[(gy - 1) * w + gx + 1];
    if (tx == 0 && ty == BLOCK_SIZE - 1 && gx > 0 && gy < h - 1)
        tile[BLOCK_SIZE + 1][0] = in[(gy + 1) * w + gx - 1];
    if (tx == BLOCK_SIZE - 1 && ty == BLOCK_SIZE - 1 &&
        gx < w - 1 && gy < h - 1)
        tile[BLOCK_SIZE + 1][BLOCK_SIZE + 1] = in[(gy + 1) * w + gx + 1];

    __syncthreads();

    // Skip border threads
    if (x >= w || y >= h) return;

    // Sobel filter (pakai tile)
    int ver =
        tile[ty + 2][tx] + 2 * tile[ty + 2][tx + 1] + tile[ty + 2][tx + 2] -
        (tile[ty][tx] + 2 * tile[ty][tx + 1] + tile[ty][tx + 2]);
    int hor =
        tile[ty][tx + 2] + 2 * tile[ty + 1][tx + 2] + tile[ty + 2][tx + 2] -
        (tile[ty][tx] + 2 * tile[ty + 1][tx] + tile[ty + 2][tx]);
    int g = (int)sqrtf(ver * ver + hor * hor);

    if (mode == 0) {
        out[y * w + x] = min(g, 255);
    } else if (mode == 1) {
        out[y * w + x] = (g > d_thresholds[0]) ? 0 : 255;
    } else {
        int bins = mode + 1;
        int idx = 0;
        while (idx < mode && g > d_thresholds[idx]) idx++;
        out[y * w + x] = (255 * idx) / (bins - 1);
    }
}