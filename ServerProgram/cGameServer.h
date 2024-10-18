#pragma once
#include <Common.h>
#include <unordered_map>

#ifdef _DEBUG
#pragma comment(lib, "Debug/Core.lib")
#else
#pragma comment(lib, "Release/Core.lib")
#endif

class cGameServer : public server_interface<GameMsg>
{
public:
	cGameServer(uint16_t _port);
	~cGameServer();

	unordered_map<uint32_t, sPlayerDescription> m_mapPlayerRoster;
	vector<uint32_t> m_vGarbageIDs;

protected:
	bool OnClientConnect(shared_ptr<connection<GameMsg>> client) override
	{
		return true;
	}

	void OnClientValidated(shared_ptr<connection<GameMsg>> client) override
	{
		message<GameMsg> msg;
		msg.header.id = GameMsg::Client_Accepted;
		client->Send(msg);
	}

	void OnClientDisconnect(shared_ptr<connection<GameMsg>> client) override
	{
		if (client)
		{
			if (m_mapPlayerRoster.find(client->GetID()) == m_mapPlayerRoster.end())
			{

			}
			else
			{
				auto& pd = m_mapPlayerRoster[client->GetID()];
				cout << "[UNGRACEFUL REMOVAL]:" + std::to_string(pd.nUniqueID) + "\n";
				m_mapPlayerRoster.erase(client->GetID());
				m_vGarbageIDs.push_back(client->GetID());
			}
		}
	}

	void OnMessage(shared_ptr<connection<GameMsg>> client, message<GameMsg>& msg) override
	{
		if (!m_vGarbageIDs.empty())
		{
			for (auto pid : m_vGarbageIDs)
			{
				message<GameMsg> m;
				m.header.id = GameMsg::Game_RemovePlayer;
				m << pid;
				std::cout << "Removing " << pid << "\n";
				MessageAllClients(m);
			}
			m_vGarbageIDs.clear();
		}

		switch (msg.header.id)
		{
		case GameMsg::Client_RegisterWithServer:
		{
			sPlayerDescription desc;
			msg >> desc;
			desc.nUniqueID = client->GetID();
			m_mapPlayerRoster.insert_or_assign(desc.nUniqueID, desc);

			message<GameMsg> msgSendID;
			msgSendID.header.id = GameMsg::Client_AssignID;
			msgSendID << desc.nUniqueID;
			MessageClient(client, msgSendID);

			message<GameMsg> msgAddPlayer;
			msgAddPlayer.header.id = GameMsg::Game_AddPlayer;
			msgAddPlayer << desc;
			MessageAllClients(msgAddPlayer);

			for (const auto& player : m_mapPlayerRoster)
			{
				message<GameMsg> msgAddOtherPlayers;
				msgAddOtherPlayers.header.id = GameMsg::Game_AddPlayer;
				msgAddOtherPlayers << player.second;
				MessageClient(client, msgAddOtherPlayers);
			}

			break;
		}

		case GameMsg::Client_UnregisterWithServer:
		{
			break;
		}

		case GameMsg::Game_UpdatePlayer:
		{
			MessageAllClients(msg, client);
			break;
		}

		default:
			break;
		}
	}
};
