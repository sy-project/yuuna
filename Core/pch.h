// pch.h: 미리 컴파일된 헤더 파일입니다.
// 아래 나열된 파일은 한 번만 컴파일되었으며, 향후 빌드에 대한 빌드 성능을 향상합니다.
// 코드 컴파일 및 여러 코드 검색 기능을 포함하여 IntelliSense 성능에도 영향을 미칩니다.
// 그러나 여기에 나열된 파일은 빌드 간 업데이트되는 경우 모두 다시 컴파일됩니다.
// 여기에 자주 업데이트할 파일을 추가하지 마세요. 그러면 성능이 저하됩니다.

#ifndef PCH_H
#define PCH_H

#ifdef COMPILE_DLL
#define CORE_DLL __declspec(dllexport)
#else
#define CORE_DLL __declspec(dllimport)
#endif // COMPILE_DLL

// 여기에 미리 컴파일하려는 헤더 추가
#include "framework.h"

extern "C"
{
	namespace Core
	{
		CORE_DLL int Init();
		CORE_DLL int End();
		namespace Log
		{
			CORE_DLL int WriteLog(std::string _str);
			CORE_DLL void SetJsonFilename(std::string& _filename);
			CORE_DLL bool LoadJson();
			CORE_DLL bool SaveJson();
			CORE_DLL std::string GetValueFromJson(const std::string& key, const std::string& defaultValue = "");
			CORE_DLL void SetValueFromJson(const std::string& key, const std::string& value);
		}
		namespace Cmake
		{
			CORE_DLL std::vector<std::string> RunCommand(const std::string& command);
		}
		namespace Lib
		{
			CORE_DLL cLibController* GetLibContInstance();
			CORE_DLL bool LoadPluginFromName(const std::string& pluginName, const std::string& dllPath);
			CORE_DLL void UnloadAllPlugin(const std::string& pluginName);
		}
		namespace Encrypt
		{
			CORE_DLL std::string EStringData(std::string _str);
			CORE_DLL void EResourceFile(const std::string& inputPath, const std::string& outputPath);
			CORE_DLL std::string EPlayerData(sPlayerDescription _data);
		}
		namespace Decrypt
		{
			CORE_DLL std::string DStringData(std::string _str);
			CORE_DLL std::vector<char> DResourceFile(const std::string& filePath);
			CORE_DLL sPlayerDescription DPlayerData(std::string _data);
		}
		namespace CONTROL
		{
			CORE_DLL void UpdateInput();
			CORE_DLL bool KeyDown(UINT key);
			CORE_DLL bool KeyUp(UINT key);
			CORE_DLL bool KeyPress(UINT key);
			CORE_DLL Vector3D MouseGet();
			CORE_DLL void MouseSet(LPARAM lParam, int WIN_X = 0, int WIN_Y = 0);
			CORE_DLL float MouseWheelGet();
			CORE_DLL void MouseWheelSet(float value);
		}
		namespace Model
		{
			CORE_DLL void Import3D(std::string path, std::string _format = "");
			CORE_DLL int GetObjIdFromName(std::string name);
			CORE_DLL std::vector<Vertex3D> GetVertex(int objId);
		}
	}
}

#endif //PCH_H
