#include <iostream>
#include <fstream>
#include <string>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: json_reader <json_file> <key>" << std::endl;
        return 1;
    }

    std::string jsonFile = argv[1];
    std::string key = argv[2];

    std::ifstream file(jsonFile);
    if (!file) {
        std::cerr << "Failed to open JSON file: " << jsonFile << std::endl;
        return 1;
    }

    json config;
    file >> config;

    if (config.contains(key)) {
        std::cout << config[key] << std::endl;  // 값을 출력하여 CMake가 사용할 수 있도록 함
        return 0;
    }
    else {
        std::cerr << "Key not found in JSON: " << key << std::endl;
        return 1;
    }
}
