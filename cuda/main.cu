#include "header/image.h"
#include "header/cuda.h"
#include "header/io.h"
#include <iostream>
#include <vector>
#include <chrono>

int main(int argc,char*argv[]){
    if(argc<6){
        std::cerr<<"Usage: ./main [cuda_type] [block_size] n input.jpg output.jpg > output.txt\n";
        std::cout << "[cuda_type] raw shared\n";
        std::cout << "[block_size] s for static (16x16), d for dynamic\n";
        return 1;
    }
    
    std::string type = argv[1];
    std::string bs = argv[2];

    int n = std::stoi(argv[3]);
    std::string inputFile  = argv[4];
    std::string outputFile = argv[5];

    unsigned char cuda_type = 'x';
    if (type == "raw" && bs == "s") {
        cuda_type = 's';
    } else if (type == "raw" && bs == "d") {
        cuda_type = 'r';
    } else if (type == "shared" && bs == "s") {
        cuda_type = 'm';
    } else if (type == "shared" && bs == "d") {
        cuda_type = 'd';
    } else {
        std::cerr<<"Usage: ./main [cuda_type] [block_size] n input.jpg output.jpg > output.txt\n";
        std::cout << "[cuda_type] raw shared\n";
        std::cout << "[block_size] s for static (16x16), d for dynamic\n";
        return 1;
    }

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
    
    // Find best config
    dim3 dim_grid, dim_block;
    if (cuda_type=='r' || cuda_type=='d')
    get_optimal_config(img.w-2, img.h-2, &dim_grid, &dim_block);
    
    auto t1 = std::chrono::high_resolution_clock::now();
    sobel(&img, mode, thresholds.data(), &dim_grid, &dim_block, cuda_type);
    
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
    deleteImage(&img);
}