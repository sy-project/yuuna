#pragma once
#include "../json/nlohmann/json.hpp"

class cJsonController : public Singleton<cJsonController>
{
	friend class Singleton;
public:
	void SetFilename(const std::string& filename);

	bool Load();
	bool Save() const;

	std::string GetValue(const std::string& key, const std::string& defaultValue = "") const;
	void SetValue(const std::string& key, const std::string& value);

private:
	std::string filename_;
	std::unordered_map<std::string, std::string> data_;

private:
	cJsonController();
	~cJsonController();
};

