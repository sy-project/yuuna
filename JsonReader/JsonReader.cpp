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
    try {
        file >> config;
    }
    catch (json::parse_error& e) {
        std::cerr << "JSON parsing error: " << e.what() << std::endl;
        return 1;
    }

    if (config.contains(key)) {
        if (config[key].is_string()) {
            std::string value = config[key].get<std::string>();
            if (value.length() > 2) {
                std::string modifiedValue = value.substr(0, value.length());
                std::cout << modifiedValue << std::endl;
            }
            else {
                std::cerr << "Value is too short to trim: " << value << std::endl;
            }
        }
        else {
            std::cerr << "Key does not contain a string value: " << key << std::endl;
        }
    }
    else {
        std::cerr << "Key not found in JSON: " << key << std::endl;
    }
}
