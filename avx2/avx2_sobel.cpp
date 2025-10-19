// File: avx2.cpp
// Implementasi Sobel paralel menggunakan AVX2

#include "Avx2_sobel.hpp" 
#include <stdexcept>
#include <cmath>


cv::Mat loadJPG(const std::string& f) {
    cv::Mat img = cv::imread(f, cv::IMREAD_GRAYSCALE);
    if(img.empty()) throw std::runtime_error("Failed Load Image: " + f);
    return img;
}

void saveJPG(const cv::Mat& img, const std::string& f) {
    if (!cv::imwrite(f, img)) {
        throw std::runtime_error("Failed Save Image: " + f);
    }
}

inline __m256 uc8_to_ps(const uchar* p) {
    __m128i uc_v_128 = _mm_loadu_si64(p);
    __m256i i_v = _mm256_cvtepu8_epi32(uc_v_128);
    return _mm256_cvtepi32_ps(i_v);
}

cv::Mat sobel_avx2(const cv::Mat& in_img, int mode, const std::vector<int>& threshold) {
    int w = in_img.cols;
    int h = in_img.rows;
    cv::Mat out_img = cv::Mat::zeros(h, w, CV_8UC1);

    // templating
    const __m256 v_zero = _mm256_setzero_ps();
    const __m256 v_255 = _mm256_set1_ps(255.0f);
    
    std::vector<int> levels;
    if (mode >= 2) {
        int k = threshold.size();
        for(int i = 0; i <= k; i++) levels.push_back((255 * i) / k);
    }

    for (int y = 1; y < h - 1; ++y) {
        const uchar* p_prev = in_img.ptr<uchar>(y - 1);
        const uchar* p_curr = in_img.ptr<uchar>(y);
        const uchar* p_next = in_img.ptr<uchar>(y + 1);
        uchar* p_out = out_img.ptr<uchar>(y);

        int x = 1;
        int w_bound = (w - 1) - ((w - 1) % 8);

        for (; x < w_bound; x += 8) {

            // paralelisasi konvolusi
            __m256 p0, p1, p2, p3, p5, p6, p7, p8;
            p0 = uc8_to_ps(p_prev + x - 1); p1 = uc8_to_ps(p_prev + x); p2 = uc8_to_ps(p_prev + x + 1);
            p3 = uc8_to_ps(p_curr + x - 1); p5 = uc8_to_ps(p_curr + x + 1);
            p6 = uc8_to_ps(p_next + x - 1); p7 = uc8_to_ps(p_next + x); p8 = uc8_to_ps(p_next + x + 1);
            
            __m256 sx = _mm256_add_ps(_mm256_mul_ps(p0, _mm256_set1_ps(-1.0f)), _mm256_mul_ps(p2, _mm256_set1_ps(1.0f)));
            sx = _mm256_add_ps(sx, _mm256_mul_ps(p3, _mm256_set1_ps(-2.0f)));
            sx = _mm256_add_ps(sx, _mm256_mul_ps(p5, _mm256_set1_ps(2.0f)));
            sx = _mm256_add_ps(sx, _mm256_mul_ps(p6, _mm256_set1_ps(-1.0f)));
            sx = _mm256_add_ps(sx, _mm256_mul_ps(p8, _mm256_set1_ps(1.0f)));

            __m256 sy = _mm256_add_ps(_mm256_mul_ps(p0, _mm256_set1_ps(1.0f)), _mm256_mul_ps(p1, _mm256_set1_ps(2.0f)));
            sy = _mm256_add_ps(sy, _mm256_mul_ps(p2, _mm256_set1_ps(1.0f)));
            sy = _mm256_add_ps(sy, _mm256_mul_ps(p6, _mm256_set1_ps(-1.0f)));
            sy = _mm256_add_ps(sy, _mm256_mul_ps(p7, _mm256_set1_ps(-2.0f)));
            sy = _mm256_add_ps(sy, _mm256_mul_ps(p8, _mm256_set1_ps(-1.0f)));

            __m256 g_v = _mm256_sqrt_ps(_mm256_add_ps(_mm256_mul_ps(sx, sx), _mm256_mul_ps(sy, sy)));

            // proses sesuai mode
            __m256 res_v;
            if (mode == 0) {
                res_v = _mm256_min_ps(g_v, v_255);
            } else if (mode == 1) {
                __m256 mask = _mm256_cmp_ps(g_v, _mm256_set1_ps((float)threshold[0]), _CMP_GT_OQ);
                res_v = _mm256_blendv_ps(v_255, v_zero, mask);
            } else { 
                res_v = _mm256_set1_ps((float)levels[0]);
                for (size_t i = 0; i < threshold.size(); ++i) {
                    __m256 mask = _mm256_cmp_ps(g_v, _mm256_set1_ps((float)threshold[i]), _CMP_GT_OQ);
                    res_v = _mm256_blendv_ps(res_v, _mm256_set1_ps((float)levels[i+1]), mask);
                }
            }
            
           // float -> int32
            __m256i g_int = _mm256_cvtps_epi32(res_v);

            // standardisasi, pastiin only in range [0,255]
            __m256i zero = _mm256_setzero_si256();
            __m256i maxv = _mm256_set1_epi32(255);
            g_int = _mm256_min_epi32(_mm256_max_epi32(g_int, zero), maxv);

            // pecah ke tmp32 array buat akses satu per satu
            alignas(32) int32_t tmp32[8];
            _mm256_storeu_si256((__m256i*)tmp32, g_int);

            // cast tiap elemen ke uchar, simpen ke output
            for (int i = 0; i < 8; ++i) {
                p_out[x + i] = static_cast<uchar>(tmp32[i] & 0xFF);
            }

        }

        // proses sisanya serial
        for (; x < w - 1; ++x) {
            int Gx_val = 0, Gy_val = 0;
            int Gx_kernel[3][3]={{-1,0,1},{-2,0,2},{-1,0,1}};
            int Gy_kernel[3][3]={{1,2,1},{0,0,0},{-1,-2,-1}};

            for(int ky = -1; ky <= 1; ky++){
                for(int kx = -1; kx <= 1; kx++){
                    uchar p_val = in_img.at<uchar>(y + ky, x + kx);
                    Gx_val += p_val * Gx_kernel[ky+1][kx+1];
                    Gy_val += p_val * Gy_kernel[ky+1][kx+1];
                }
            }
            
            float g = std::sqrt((float)Gx_val*Gx_val + (float)Gy_val*Gy_val);
            
            if (mode == 0) p_out[x] = (uchar)std::min(g, 255.0f);
            else if (mode == 1) p_out[x] = (g > threshold[0]) ? 0 : 255;
            else {
                int l_idx = 0;
                while(l_idx < (int)threshold.size() && g > threshold[l_idx]) l_idx++;
                p_out[x] = (uchar)levels[l_idx];
            }
        }
    }
    return out_img;
}

