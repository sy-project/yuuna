#pragma once

class cNetworkManager : public Singleton<cNetworkManager>, client_interface<GameMsg>
{
private:
	friend class Singleton;

private:
	string serverip;
	int serverport;

	unordered_map<uint32_t, sPlayerDescription> mapObjects;
	uint32_t nPlayerID = 0;
	sPlayerDescription descPlayer;

	bool bWaitingForConnection = true;

public:
	void Update();

private:
	void SetIdle(int);

public:
	void SetServer(string _ip, int _port) { serverip = _ip; serverport = _port; }
	bool ConServer();

private:
	cNetworkManager();
	~cNetworkManager();
};

