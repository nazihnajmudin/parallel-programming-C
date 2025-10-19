// File: Avx2.hpp
// File ini adalah header untuk program Avx2. Deklarasi fungsi yang ada pada program avx_sobel
#ifndef AVX2_HPP
#define AVX2_HPP

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <immintrin.h> 

cv::Mat loadJPG(const std::string& f);
/*
  loadJPG -> load file JPG ke format OpenCV
  - Input : const f (filename) berisi nama file 
  - Output: format opencv berupa cv::Mat
*/

void saveJPG(const cv::Mat& img, const std::string& f);
/*
  saveJPG -> fungsi yang digunakan untuk cetak gambar dari cv:Mat image ke jpg di f (filename)
*/

inline __m256 uc8_to_ps(const uchar* p);
/*
  uc8_to_ps -> fungsi inline untuk mengkonversi 8 uchar ke dalam __m256 (float)
  - Input : const uchar* p (pointer ke array uchar)
  - Output: __m256 (array float 8 elemen)
*/  

cv::Mat sobel_avx2(const cv::Mat& img_input, int mode, const std::vector<int>& thresholds);
/*
  sobel_avx2 -> main function yang melakukan operasi sobel menggunakan paralelisasi avx2
*/

#endif 