#pragma once
#include <iostream>
#include <filesystem>
#include <string>

std::string getExtension(const std::string& path) {
    std::filesystem::path p(path);
    std::string ext = p.extension().string();
    if (!ext.empty() && ext[0] == '.')
        ext = ext.substr(1);
    return ext;
}

std::string GetFileName(const std::string& path) {
    std::filesystem::path filePath(path);
    return filePath.stem().string();
}

std::string GetFilePath(const std::string& path)
{
    std::filesystem::path filePath(path);
    return filePath.parent_path().string();
}