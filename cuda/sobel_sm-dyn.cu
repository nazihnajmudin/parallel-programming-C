#include "header/cuda.h"
#include "header/image.h"

__global__ void kernel_sobel_tiled(const unsigned char* __restrict__ in, unsigned char* out,
                        int w, int h, int mode, const int* d_thresholds) {
    const int BX = blockDim.x;
    const int BY = blockDim.y;
    const int sW = BX + 2;   // shared width (cols)
    const int sH = BY + 2;   // shared height (rows)
    extern __shared__ unsigned char s[]; // size: sW * sH

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int x = bx * BX + tx;  // global x for this thread
    int y = by * BY + ty;  // global y for this thread

    int tid = ty * BX + tx;
    int totalThreads = BX * BY;
    int elems = sW * sH;

    // Linearized load (each thread loads multiple elements of shared tile)
    for (int i = tid; i < elems; i += totalThreads) {
        int sm_y = i / sW;
        int sm_x = i % sW;
        int glob_x = bx * BX + (sm_x - 1); // -1 because sm_x=0 is left
        int glob_y = by * BY + (sm_y - 1); // -1 because sm_y=0 is top

        // clamp to image border
        glob_x = (glob_x < 0) ? 0 : ( (glob_x >= w) ? (w-1) : glob_x );
        glob_y = (glob_y < 0) ? 0 : ( (glob_y >= h) ? (h-1) : glob_y );

        s[i] = in[glob_y * w + glob_x];
    }

    __syncthreads();

    if (x >= w || y >= h) return;

    // center in shared coords
    int c_x = tx + 1;
    int c_y = ty + 1;
    // int idx_center = c_y * sW + c_x;

    int ver =
        s[(c_y+1)*sW + (c_x-1)] + 2*s[(c_y+1)*sW + c_x] + s[(c_y+1)*sW + (c_x+1)]
      - (s[(c_y-1)*sW + (c_x-1)] + 2*s[(c_y-1)*sW + c_x] + s[(c_y-1)*sW + (c_x+1)]);
    int hor =
        s[(c_y-1)*sW + (c_x+1)] + 2*s[c_y*sW + (c_x+1)] + s[(c_y+1)*sW + (c_x+1)]
      - (s[(c_y-1)*sW + (c_x-1)] + 2*s[c_y*sW + (c_x-1)] + s[(c_y+1)*sW + (c_x-1)]);

    int g = (int)sqrtf((float)(ver*ver + hor*hor));

    // write result
    if (mode == 0) {
        out[y*w + x] = (unsigned char) min(g, 255);
    } else if (mode == 1) {
        out[y*w + x] = (g > d_thresholds[0]) ? 0 : 255;
    } else {
        int bins = mode + 1;
        int idx = 0;
        while (idx < mode && g > d_thresholds[idx]) idx++;
        out[y*w + x] = (255 * idx) / (bins - 1);
    }
}