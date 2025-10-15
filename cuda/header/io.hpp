#ifndef IO_HPP
#define IO_HPP

#include "image.h"
#include <string>

image_t loadJPG(const std::string &f);

void saveJPG(const image_t &img, const std::string &f);

#endif