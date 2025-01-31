#include "pch.h"
#include "cJsonController.h"

using json = nlohmann::json;

void cJsonController::SetFilename(const std::string& filename)
{
    filename_ = filename;
}

bool cJsonController::Load()
{
    std::ifstream file(filename_);
    if (!file.is_open()) {
        std::cerr << "Failed to open JSON file: " << filename_ << std::endl;
        return false;
    }

    try {
        json j;
        file >> j;
        for (auto& [key, value] : j.items()) {
            data_[key] = value.get<std::string>();
        }
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Error parsing JSON: " << e.what() << std::endl;
        return false;
    }
}

bool cJsonController::Save() const
{
    std::ofstream file(filename_);
    if (!file.is_open()) {
        std::cerr << "Failed to open JSON file for writing: " << filename_ << std::endl;
        return false;
    }

    try {
        json j;
        for (const auto& [key, value] : data_) {
            j[key] = value;
        }
        file << j.dump(4); // 4칸 들여쓰기하여 저장
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Error writing JSON: " << e.what() << std::endl;
        return false;
    }
}

std::string cJsonController::GetValue(const std::string& key, const std::string& defaultValue) const
{
    if (data_.find(key) != data_.end()) {
        return data_.at(key);
    }
    return defaultValue;
}

void cJsonController::SetValue(const std::string& key, const std::string& value)
{
    data_[key] = value;
}

cJsonController::cJsonController()
{
}

cJsonController::~cJsonController()
{
}
