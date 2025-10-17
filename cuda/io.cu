#include "header/image.h"
#include "header/io.h"

#include <opencv2/opencv.hpp>

image_t loadJPG(const std::string &f) {
    cv::Mat mat = cv::imread(f, cv::IMREAD_GRAYSCALE);
    if (mat.empty()) {
        throw std::runtime_error("Failed to load image");
    }

    image_t img;
    createImage(&img, mat.cols, mat.rows);

    for (int y = 0; y < img.h; y++) {
        for (int x = 0; x < img.w; x++) {
            PIXEL(img, x, y) = mat.at<uchar>(y, x);
        }
    }
    return img;
}

void saveJPG(const image_t &img, const std::string &f) {
    cv::Mat mat(img.h, img.w, CV_8UC1);
    for (int y = 0; y < img.h; y++) {
        for (int x = 0; x < img.w; x++) {
            mat.at<uchar>(y, x) = PIXEL(img, x, y);
        }
    }
    cv::imwrite(f, mat);
}