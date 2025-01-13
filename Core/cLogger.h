
class cLogger : public Singleton<cLogger>
{
	friend class Singleton;
public:
	void Logging(std::string _str);

private:
	cLogger();
	~cLogger();
};
