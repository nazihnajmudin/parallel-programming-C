#include "header/cuda.h"
#include "header/image.h"

__global__ void kernel_sobel_raw(unsigned char *in, unsigned char *out, int w, int h, int mode, int *d_thresholds) {
    // Cara indexing gambar sudah sesuai dengan indexing di cuda
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    x++; y++;

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