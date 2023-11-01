#pragma once
class cVector3
{
public:
	Vector4 data;

	cVector3();
	cVector3(Float3 value);
	cVector3(float x, float y, float z);
	cVector3(Vector4 value);

	operator Float3();

	void SetX(float value);
	void SetY(float value);
	void SetZ(float value);
	void SetW(float value);
	float GetX();
	float GetY();
	float GetZ();
	float GetW();

	_declspec(property(get = GetX, put = SetX)) float x;
	_declspec(property(get = GetY, put = SetY)) float y;
	_declspec(property(get = GetZ, put = SetZ)) float z;
	_declspec(property(get = GetW, put = SetW)) float w;

	cVector3 operator+(const cVector3& value) const;
	cVector3 operator-(const cVector3& value) const;
	cVector3 operator*(const cVector3& value) const;
	cVector3 operator/(const cVector3& value) const;

	void operator+=(const cVector3& value);
	void operator-=(const cVector3& value);
	void operator*=(const cVector3& value);
	void operator/=(const cVector3& value);

	cVector3 operator+(const float& value) const;
	cVector3 operator-(const float& value) const;
	cVector3 operator*(const float& value) const;
	cVector3 operator/(const float& value) const;

	void operator+=(const float& value);
	void operator-=(const float& value);
	void operator*=(const float& value);
	void operator/=(const float& value);

	friend cVector3 operator+(const float value1, const cVector3& value2);
	friend cVector3 operator-(const float value1, const cVector3& value2);
	friend cVector3 operator*(const float value1, const cVector3& value2);
	friend cVector3 operator/(const float value1, const cVector3& value2);

	friend cVector3 operator+(const Float3& value1, const cVector3& value2);
	friend cVector3 operator-(const Float3& value1, const cVector3& value2);
	friend cVector3 operator*(const Float3& value1, const cVector3& value2);
	friend cVector3 operator/(const Float3& value1, const cVector3& value2);

	bool operator== (const cVector3& value) const;

	float operator[](const UINT& index) const;

	float Length() const;

	cVector3 Normal() const;
	void Normalize();

	static cVector3 Cross(const cVector3& vec1, const cVector3& vec2);
	static float Dot(const cVector3& vec1, const cVector3& vec2);
};

