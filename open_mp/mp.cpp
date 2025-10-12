// mp.cpp - Implementasi fungsi-fungsi untuk deteksi tepi Sobel menggunakan OpenMP

#include <vector>
#include <cmath>
#include <stdexcept>
#include <opencv2/opencv.hpp>
#include "header/Image.hpp"

using namespace std;

// muat gambar grayscale pake open mp
Image loadJPG(const string &f) {
    // return gambar 1-channel
    cv::Mat mat = cv::imread(f, cv::IMREAD_GRAYSCALE);
    if(mat.empty()) throw runtime_error("Gagal: " + f);
    Image img{mat.cols, mat.rows};
    // salin data pixel dari opencv ke struct Image
    for(int r=0; r<mat.rows; ++r)
        for(int c=0; c<mat.cols; ++c)
            img.at(r,c) = mat.at<uchar>(r,c);
    return img;
}

// nyimpen gambar grayscale pake openmp
void saveJPG(const Image &img, const string &f) {
    cv::Mat mat(img.height, img.width, CV_8UC1);
    // salin data pixel dari struct Image ke opencv
    for(int r=0; r<img.height; ++r)
        for(int c=0; c<img.width; ++c)
            mat.at<uchar>(r,c) = img.at(r,c);
    cv::imwrite(f, mat); // ini otomatis bikin file baru atau timpa file lama, dengan format sesuai ekstensi
}

// hitung Sobel pada gambar 'in' dengan mode dan threshold tertentu
// Mode: 0 => magnitudo gradien (dibatasi 0..255)
//       1 => binary inverted dengan threshold[0] (default 128)
//       2 => kuantisasi multi-level dengan vektor threshold

Image SOBELOpenMP(const Image &in, int mode, const vector<int> &thresholds) {
    Image out(in.width, in.height);
    
    // ini kita set pixel border (pinggir) sama dengan input
    // karena kita skip proses di pinggir (tidak ada tetangga lengkap 3x3)
    for(int r=0; r<in.height; ++r) {
        out.at(r,0) = in.at(r,0);
        out.at(r,in.width-1) = in.at(r,in.width-1);
    }
    for(int c=0; c<in.width; ++c) {
        out.at(0,c) = in.at(0,c);
        out.at(in.height-1,c) = in.at(in.height-1,c);
    }

    // hitung level untuk mode multi-threshold
    vector<int> levels;
    if(mode == 2) { // (kalo multi-level)
        int k = (int)thresholds.size(); // k itu jumlah threshold
        int bins = k + 1; // bins itu jumlah level = k+1
        levels.resize(bins); // alokasi ruang
        // bagi level secara merata di [0..255]
        // misal k=2 => bins=3 => levels={0,127,255}
        // misal k=3 => bins=4 => levels={0,85,170,255}
        for(int i=0; i<bins; i++) levels[i] = (255 * i) / (bins - 1);
    }

    // Kernel Sobel di Image.hpp => SOBEL_X dan SOBEL_Y
    // paralel loop luar pake OpenMP.
    // Setiap thread akan memproses sekumpulan baris yang berurutan (kecuali baris pertama dan terakhir). Kenapa ? karena bakal skip baris pinggir .

    // Contoh pembagian baris dengan 3 thread dan tinggi=10:
    // Thread:  0   |   1   |   2
    // Baris:  1-3  |  4-6  |  7-8   (baris 0 dan 9 adalah border yang di-skip)

    #pragma omp parallel for schedule(static) // tiap thread dapat baris berurutan
    for(int r = 1; r < in.height - 1; ++r) {
        // CATATAN: di loop ini aman kalo nulis ke out.at(r,*) karena
        // setiap r unik per iterasi. Pembacaan dari 'in' bersifat read-only.
        for(int c = 1; c < in.width - 1; ++c) {
            int sx = 0, sy = 0;
            
            // Konvolusi dengan neighborhood 3x3
            for(int kr = -1; kr <= 1; ++kr) {
                for(int kc = -1; kc <= 1; ++kc) {
                    int val = in.at(r + kr, c + kc);
                    sx += val * SOBEL_X[kr+1][kc+1];  // Gradien horizontal
                    sy += val * SOBEL_Y[kr+1][kc+1];  // Gradien vertikal
                }
            }
            // Hitung magnitudo gradien menggunakan norma Euclidean
            int G = (int)std::sqrt((double)(sx*sx + sy*sy));

            if(mode == 0) {
                // Mode 0: magnitudo gradien dibatasi hingga 255
                out.at(r,c) = (G > 255) ? 255 : G;
            } else if(mode == 1) {
                // Mode 1: binary inverted: G > T => hitam (0), selainnya putih (255)
                int T0 = thresholds.empty() ? 128 : thresholds[0];
                out.at(r,c) = (G > T0) ? 0 : 255;
            } else {
                // Mode 2: multi-level, penjelasan di spek 
                int idx = 0;
                while(idx < (int)thresholds.size() && G > thresholds[idx]) ++idx;
                out.at(r,c) = (unsigned char)levels[idx];
            }
        }
    }

    return out;
}

