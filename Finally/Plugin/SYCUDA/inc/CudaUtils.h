#pragma once
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <iostream>
uint8_t* loadImage(std::string filename, int* width, int* height) {
    int channels;
    uint8_t* imageData = stbi_load(filename.c_str(), width, height, &channels, STBI_rgb_alpha);

    if (!imageData) {
        std::cerr << "Failed to load image: " << filename << std::endl;
        return nullptr;
    }

    return imageData;
}