#pragma once

class cEncryptManager : public Singleton<cEncryptManager>
{
private:
	friend class Singleton;

public:
	void Encrypt();
	void Decrypt();

	std::string EncryptData(std::string _data);
	std::string DecryptData(std::string _data);

	std::string EncryptNetwork(sPlayerDescription _data);
	sPlayerDescription DecryptNetwork(std::string _data);

	void SetKey(int _key) { key = _key; }

private:
	int key;
	cEncryptManager();
	~cEncryptManager();
};

