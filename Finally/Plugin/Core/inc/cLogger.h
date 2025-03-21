
class cLogger : public Singleton<cLogger>
{
	template<typename T>
	friend class Singleton;
public:
	void Logging(std::string _str);

private:
	cLogger();
	~cLogger();
};
