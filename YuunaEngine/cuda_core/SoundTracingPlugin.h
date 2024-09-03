#pragma once

#include <vector>
#include <unordered_map>
#include <algorithm>
#include <memory>
#include <Singleton.h>

class SoundTracer : public Singleton<SoundTracer>
{
	friend class Singleton;
public:
	void Init();
private:
	SoundTracer();
	~SoundTracer();
};