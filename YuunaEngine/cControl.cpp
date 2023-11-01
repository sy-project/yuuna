#include "header.h"

cControl::cControl()
	: curState{}, oldState{}, mapState{}, wheelValue(0)
{
}

cControl::~cControl()
{
}

void cControl::Update()
{
	memcpy(oldState, curState, sizeof(oldState));

	//0, 1, 128, 129
	GetKeyboardState(curState);

	for (int i = 0; i < KEYMAX; i++)
	{
		unsigned char key = curState[i] & 0x80;
		curState[i] = key ? 1 : 0;

		int old = oldState[i];
		int cur = curState[i];

		if (old == 0 && cur == 1)
			mapState[i] = DOWN;
		else if (old == 1 && cur == 0)
			mapState[i] = UP;
		else if (old == 1 && cur == 1)
			mapState[i] = PRESS;
		else
			mapState[i] = NONE;
	}
}

void cControl::SetMouse(LPARAM lParam)
{
	float x = (float)LOWORD(lParam);
	float y = (float)HIWORD(lParam);

	if (x <= WINDOW_WIDTH)
		mousePos.x = x;
	if (y <= WINDOW_HEIGHT)
		mousePos.y = y;
}
