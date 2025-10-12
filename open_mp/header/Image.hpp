#ifndef IMAGE_HPP
#define IMAGE_HPP

#include <vector>
#include <cstdint>

using pixel_t = uint8_t;

// nyimpen gambar grayscale
struct Image {
    int width{0}, height{0};
    std::vector<pixel_t> pixels; // data pixel: width*height

    // Konstruktor
    Image() = default;
    Image(int w, int h): width(w), height(h), pixels(w*h) {}

    // akses pixel baris r, kolom c
    pixel_t &at(int r, int c) { return pixels[r*width + c]; } // ini buat non-constnya, biar bisa dipake di fungsi yang non-const
    const pixel_t &at(int r, int c) const { return pixels[r*width + c]; } // ini buat constnya, biar bisa dipake di fungsi yang const
};
inline constexpr int SOBEL_X[3][3] = {
    { -1, 0, 1 },    
    { -2, 0, 2 },    
    { -1, 0, 1 }
};

inline constexpr int SOBEL_Y[3][3] = {
    {  1,  2,  1 },
    {  0,  0,  0 },
    { -1, -2, -1 }
};

#endif