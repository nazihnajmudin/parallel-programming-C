#include "header/image.h"
#include "header/cuda.h"
#include "header/io.hpp"
#include <iostream>
#include <vector>
#include <chrono>

int main(int argc,char*argv[]){
    if(argc<4){
        std::cerr<<"Usage: ./main n input.jpg output.jpg > output.txt\n";
        return 1;
    }

    int n = std::stoi(argv[1]);
    std::string inputFile  = argv[2];
    std::string outputFile = argv[3];

    if (n < 0) {
        std::cerr<<"Error: n must be greater than or equal to 0\n";
        return 1;
    }

    
    int mode;
    std::vector<int> thresholds;
    
    if (n == 0) {
        mode = 0; // gradient magnitude
    } else if (n == 1) {
        mode = 1; // Binary threshold
        thresholds.push_back(128);
    } else {
        mode = n; // Multi-level thresholds
        for (int i = 1; i < n; i++) {
            int threshold = (255 * i) / n;
            thresholds.push_back(threshold);
        }
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    image_t img = loadJPG(inputFile);
    auto t1 = std::chrono::high_resolution_clock::now();
    sobel(&img, mode, thresholds.data());
    auto t2 = std::chrono::high_resolution_clock::now();
    saveJPG(img, outputFile);
    auto t3 = std::chrono::high_resolution_clock::now();

    auto tInput  = std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count();
    auto tProc   = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();
    auto tOutput = std::chrono::duration_cast<std::chrono::milliseconds>(t3-t2).count();

    std::cout << "\n================ Sobel Edge Detection ================\n";
    std::cout << "Program Type : CUDA\n";
    std::cout << "------------------------------------------------------\n";
    std::cout << "Mode         : " << mode << "\n";
    std::cout << "Image size   : w = " << img.w << " h = " << img.h << "\n";
    if (!thresholds.empty()) {
        std::cout << "Threshold(s) : ";
        for (auto t : thresholds) std::cout << t << " ";
        std::cout << "\n";
    }
    std::cout << "------------------------------------------------------\n";
    std::cout << "Timing (ms)\n";
    std::cout << "  Input      : " << tInput  << "\n";
    std::cout << "  Processing : " << tProc   << "\n";
    std::cout << "  Output     : " << tOutput << "\n";
    std::cout << "======================================================\n";

    // Bebaskan Memori
    DELETE(img);
}