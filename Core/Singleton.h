#pragma once
#ifndef SINGLETON_H
#define SINGLETON_H

template<typename T>
class Singleton
{
protected:
	static T* instance;
public:
	static T* Get()
	{
		if (instance == nullptr)
			instance = new T();
		return instance;
	}
	static void Delete()
	{
		delete instance;
	}
};
template<typename T>
T* Singleton<T>::instance = nullptr;

#endif // !SINGLETON_H
