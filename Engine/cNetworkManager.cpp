#include "framework.h"

void cNetworkManager::Update()
{
	if (IsConnected())
	{
		while (!Incoming().empty())
		{
			auto msg = Incoming().pop_front().msg;

			switch (msg.header.id)
			{
			case(GameMsg::Client_Accepted):
			{
				message<GameMsg> msg;
				msg.header.id = GameMsg::Client_RegisterWithServer;
				descPlayer.vPos = { 3.0f,3.0f,3.0f };
				msg << descPlayer;
				Send(msg);
				break;
			}
			case(GameMsg::Client_AssignID):
			{
				msg >> nPlayerID;
				break;
			}
			case(GameMsg::Game_AddPlayer):
			{
				sPlayerDescription desc;
				msg >> desc;
				mapObjects.insert_or_assign(desc.nUniqueID, desc);

				if (desc.nUniqueID == nPlayerID)
				{
					bWaitingForConnection = false;
				}
				break;
			}
			case (GameMsg::Game_RemovePlayer):
			{
				uint32_t nRemovalID = 0;
				msg >> nRemovalID;
				mapObjects.erase(nRemovalID);
				break;
			}
			case (GameMsg::Game_UpdatePlayer):
			{
				sPlayerDescription desc;
				msg >> desc;
				mapObjects.insert_or_assign(desc.nUniqueID, desc);
				break;
			}
			default:
				break;
			}
		}
	}

	if (bWaitingForConnection)
	{
		return;
	}
}

void cNetworkManager::SetIdle(int instance)
{
	if (bWaitingForConnection)
	{
		return;
	}
}

bool cNetworkManager::ConServer()
{
	if (Connect(serverip, serverport))
		return true;
	return false;
}

cNetworkManager::cNetworkManager()
{
}

cNetworkManager::~cNetworkManager()
{
}
