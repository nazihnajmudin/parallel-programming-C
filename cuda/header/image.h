#ifndef IMAGE_H
#define IMAGE_H

/* IMAGE STRUCT */
typedef struct image_t {
    int w, h;
    unsigned char *p;
} image_t;

/* PIXEL ACCESS */
#define pixel(img, x, y) img->p[y * img->w + x]
#define PIXEL(img, x, y) img.p[y * img.w + x]

/* FREE MEMORY */
#define FREE(img) free(img.p)       // free malloc C
#define DELETE(img) delete[] img.p  // delete new C++

/* DIRECT CUDA ACCESS */
void createImage(image_t *img, int w, int h);

void deleteImage(image_t *img);

#endif