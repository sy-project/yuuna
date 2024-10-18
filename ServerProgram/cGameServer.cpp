#include "cGameServer.h"

cGameServer::cGameServer(uint16_t nPort) : server_interface<GameMsg>(nPort)
{
}

cGameServer::~cGameServer()
{
}
