#pragma once
#ifdef ONLY_ENGINE
#include "json/nlohmann/json.hpp"
#else
#include "../json/nlohmann/json.hpp"
#endif // ONLY_ENGINE

class cJsonController : public Singleton<cJsonController>
{
	template<typename T>
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

