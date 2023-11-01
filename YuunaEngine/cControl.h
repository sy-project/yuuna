#pragma once

#include "cVector3.h"

#define KEYMAX 255

class cControl : public Singleton<cControl>
{
private:
	friend class Singleton;

	enum {
		NONE,
		DOWN,
		UP,
		PRESS
	};

	unsigned char curState[KEYMAX];
	unsigned char oldState[KEYMAX];
	unsigned char mapState[KEYMAX];

	cVector3 mousePos;
	float wheelValue;

	cControl();
	~cControl();

public:
	void Update();

	bool Down(UINT key) { return mapState[key] == DOWN; }
	bool Up(UINT key) { return mapState[key] == UP; }
	bool Press(UINT key) { return mapState[key] == PRESS; }

	cVector3 GetMouse() { return mousePos; }
	void SetMouse(LPARAM lParam);

	float GetWheel() { return wheelValue; }
	void SetWheel(float value) { wheelValue = value; }
};

