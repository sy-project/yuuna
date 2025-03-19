#pragma once

using FunctionPtr = void(*)();

struct PluginInfo {
	HMODULE handle;
	std::unordered_map<std::string, FunctionPtr> functions;
};

class cLibController : public Singleton<cLibController>
{
	friend class Singleton;
public:
	cLibController(const cLibController&) = delete;
	cLibController& operator=(const cLibController&) = delete;

	bool LoadPlugin(const std::string& pluginName, const std::string& dllPath);

	template<typename FuncType>
	FuncType GetFunction(const std::string& pluginName, const std::string& functionName) {
		if (loadedPlugins.find(pluginName) == loadedPlugins.end()) {
			return nullptr;
		}
		auto& plugin = loadedPlugins[pluginName];
		if (plugin.functions.find(functionName) == plugin.functions.end()) {
			return nullptr;
		}
		return reinterpret_cast<FuncType>(plugin.functions[functionName]);
	}
	void UnloadPlugin(const std::string& pluginName);
	const std::unordered_map<std::string, PluginInfo>& GetLoadedPlugins() const {
		return loadedPlugins;
	}
private:
	std::unordered_map<std::string, PluginInfo> loadedPlugins;

private:
	cLibController();
	~cLibController();
};

