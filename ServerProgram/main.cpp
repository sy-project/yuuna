#include "cGameServer.h"

int main()
{
	cGameServer server(60000);
	server.Start();

	while (1)
	{
		server.Update(-1, true);
	}

	return 0;
}