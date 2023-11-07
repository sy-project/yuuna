#pragma once

namespace Utility
{
	string ToString(wstring value);
	wstring ToWString(string value);

	void Replace(string* str, string comp, string rep);
	vector<string> SplitString(string origin, string tok);

	wstring GetExtension(wstring path);
	string GetDirectoryName(string path);
	string GetFileName(string path);
	string GetFileNameWithoutExtension(string path);

	void CreateFolders(string path);

	bool ExistDirectory(string path);
	bool ExistFile(string path);
}