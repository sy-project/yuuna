#pragma once
#include "Types.h"

#define KEYMAX 255
class cControl : public Singleton<cControl>
{
	friend class Singleton;
public:
	void Update();

	bool Down(UINT key) { return mapState[key] == DOWN; }
	bool Up(UINT key) { return mapState[key] == UP; }
	bool Press(UINT key) { return mapState[key] == PRESS; }

	Vector3D GetMouse() { return mousePos; }
	void SetMouse(LPARAM lParam, int WIN_X = 0, int WIN_Y = 0);

	float GetWheel() { return wheelValue; }
	void SetWheel(float value) { wheelValue = value; }

private:
	enum {
		NONE,
		DOWN,
		UP,
		PRESS
	};

	unsigned char curState[KEYMAX];
	unsigned char oldState[KEYMAX];
	unsigned char mapState[KEYMAX];

	Vector3D mousePos;
	float wheelValue;

	cControl();
	~cControl();
};

