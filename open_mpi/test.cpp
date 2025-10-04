#include <opencv2/opencv.hpp>
#include <iostream>
#include <mpi.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    std::cout << "OpenCV version: " << CV_VERSION << std::endl;
    cv::Mat img = cv::Mat::zeros(100, 100, CV_8UC3);
    cv::imshow("Test", img);
    cv::waitKey(0);
    MPI_Finalize();
    return 0;
}