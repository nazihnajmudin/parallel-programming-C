// main.cpp - Program utama untuk OpenMP Sobel edge detection
// Penggunaan: ./mp n input.jpg output.jpg > output.txt

#include <iostream>
#include <vector>
#include <chrono>
#include <string>
#include "header/Image.hpp"
#include "header/OpenMP.hpp"

using namespace std;
using clk = chrono::high_resolution_clock;

Image loadJPG(const string &f);
void saveJPG(const Image &img, const string &f);
Image SOBELOpenMP(const Image &in, int mode, const vector<int> &thresholds);

int main(int argc, char* argv[]) {
    if(argc < 4) {
        cerr << "Penggunaan: ./mp n input.jpg output.jpg > output.txt\n";
        return 1;
    }

    int n = stoi(argv[1]);
    string inputFile = argv[2];
    string outputFile = argv[3];

    if(n < 0) {
        cerr << "Error: n harus >= 0\n";
        return 1;
    }

    int mode;
    vector<int> thresholds;

    // Tentukan mode dan threshold berdasarkan nilai n
    if(n == 0) {
        mode = 0;  // Mode gradient magnitude
    } else if(n == 1) {
        mode = 1;  // Mode binary threshold
        thresholds.push_back(128);
    } else {
        mode = 2;  // Mode multi-level quantization
        // Untuk mode n (n>1) kita generate (n-1) threshold yang terdistribusi merata
        // antara (0,255), sesuai contoh serial: threshold: 255*i/n untuk i=1..n-1
        for(int i = 1; i < n; ++i) thresholds.push_back((255 * i) / n);
    }

    // Pengukuran waktu internal - dipisah menjadi 3 bagian sesuai spesifikasi tugas
    auto t0 = clk::now();
    Image img = loadJPG(inputFile);        // Waktu Input
    auto t1 = clk::now();
    Image res = SOBELOpenMP(img, mode, thresholds);  // Waktu Processing
    auto t2 = clk::now();
    saveJPG(res, outputFile);             // Waktu Output
    auto t3 = clk::now();

    auto tInput  = chrono::duration_cast<chrono::milliseconds>(t1 - t0).count();
    auto tProc   = chrono::duration_cast<chrono::milliseconds>(t2 - t1).count();
    auto tOutput = chrono::duration_cast<chrono::milliseconds>(t3 - t2).count();

    // Cetak laporan ringkasan ke stdout (akan dialihkan ke output.txt)
    cout << "================ Sobel Edge Detection ================\n";
    cout << "Program Type : OpenMP (shared-memory parallel)\n";
    cout << "------------------------------------------------------\n";
    cout << "Mode         : " << mode << "\n";
    if(!thresholds.empty()) {
        cout << "Threshold(s) : ";
        for(auto t : thresholds) cout << t << " ";
        cout << "\n";
    }
    cout << "------------------------------------------------------\n";
    cout << "Timing (ms)\n";
    cout << "  Input      : " << tInput  << "\n";
    cout << "  Processing : " << tProc   << "\n";
    cout << "  Output     : " << tOutput << "\n";
    cout << "======================================================\n";

    return 0;
}