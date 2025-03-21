#pragma once

class cEncryptManager : public Singleton<cEncryptManager>
{
private:
	template<typename T>
	friend class Singleton;

public:
	void Encrypt();
	void Decrypt();

	std::string EncryptData(std::string _data);
	std::string DecryptData(std::string _data);

	void EncryptResourceFile(const std::string& inputPath, const std::string& outputPath);
	std::vector<char> DecryptResourceFile(const std::string& filePath);

	std::string EncryptNetwork(sPlayerDescription _data);
	sPlayerDescription DecryptNetwork(std::string _data);

private:
	cEncryptManager();
	~cEncryptManager();
};

