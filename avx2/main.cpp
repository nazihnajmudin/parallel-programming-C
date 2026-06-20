#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include "Avx2_sobel.hpp" 

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::milliseconds;

int main(int argc, char** argv) {
    if (argc < 4) {
        return -1;
    }

    try {
        int mode = std::stoi(argv[1]);
        std::string input_file_path = argv[2];
        std::string output_file_path = argv[3];
        
        std::vector<int> thresholds;

        if (mode == 0) mode = 0;
        else if (mode == 1) {
            mode = 1;
            thresholds.push_back(128); // based on req
        } else {
            if (argc != 4 + mode) {
                std::cerr << "Error: Mode " << mode << " butuh " << mode << " threshold." << std::endl;
                return -1;
            }
            for (int i = 4; i < argc; ++i) thresholds.push_back(std::atoi(argv[i]));
            mode = thresholds.size();
        }
        
        // Load JPG
        auto t_start = high_resolution_clock::now();
        cv::Mat img = loadJPG(input_file_path);
        auto t_input_done = high_resolution_clock::now();

        // proses JPG (Avx2)
        cv::Mat sobel_result = sobel_avx2(img, mode, thresholds);
        auto t_process_done = high_resolution_clock::now();

        // Output file JPG
        saveJPG(sobel_result, output_file_path);
        auto t_output_done = high_resolution_clock::now();

        // Hitung durasi
        double input_time = duration_cast<milliseconds>(t_input_done - t_start).count();
        double process_time = duration_cast<milliseconds>(t_process_done - t_input_done).count();
        double output_time = duration_cast<milliseconds>(t_output_done - t_process_done).count();

        // Cetak laporan
        std::cout << "<=== Hasil Komputasi Menggunakan Avx2 ===>" << std::endl;
        std::cout << "<> Input  : " << input_file_path << std::endl;
        std::cout << "<> Output : " << output_file_path << std::endl;
        std::cout << "<> Mode   : " << mode << std::endl;
        if (!thresholds.empty()) {
            std::cout << "Thresholds : ";
            for (size_t i=0; i<thresholds.size(); ++i) std::cout << thresholds[i] << (i==thresholds.size()-1 ? "" : ", ");
            std::cout << std::endl;
        }

        std::cout << "\nWaktu Eksekusi:" << std::endl;
        std::cout << "  - Input      : " << input_time << " ms" << std::endl;
        std::cout << "  - Processing : " << process_time << " ms" << std::endl;
        std::cout << "  - Output     : " << output_time << " ms" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Terjadi error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

