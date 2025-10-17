#include "header/cuda.h"
#include "header/image.h"

/* Allocate image on both CPU and GPU */
void createImage(image_t *img, int w, int h) {
    img->w = w;
    img->h = h;
    CUDA_CHECK(cudaMallocManaged(&img->p, w*h*sizeof(unsigned char)));
}

/* delete image on both CPU and GPU */
void deleteImage(image_t *img) {
    CUDA_CHECK(cudaFree(img->p));
}