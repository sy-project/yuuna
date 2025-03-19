#pragma once

#include <vector>
#include <unordered_map>
#include <algorithm>
#include <memory>
#include <Singleton.h>

enum eFlag
{
	direct = 0x01,
	diffract = 0x02,
	reflect = 0x04,
	reverb = 0x08,
};

class SoundTracer : public Singleton<SoundTracer>
{
	friend class Singleton;
public:
	void Init();
	void UpdateListener();
private:
	SoundTracer();
	~SoundTracer();

	cudaError_t InitDev(int* c, const int* a, const int* b, unsigned int size);
};