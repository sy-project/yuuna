#include "header.h"

cVector3::cVector3()
{
	data = XMVectorZero();
}

cVector3::cVector3(Float3 value)
{
	data = XMLoadFloat3(&value);
}

cVector3::cVector3(float x, float y, float z)
{
	data = XMVectorSet(x, y, z, 0);
}

cVector3::cVector3(Vector4 value)
	: data(value)
{
}

cVector3::operator Float3()
{
	Float3 result;
	XMStoreFloat3(&result, data);

	return result;
}

void cVector3::SetX(float value)
{
	data = XMVectorSetX(data, value);
}

void cVector3::SetY(float value)
{
	data = XMVectorSetY(data, value);
}

void cVector3::SetZ(float value)
{
	data = XMVectorSetZ(data, value);
}

void cVector3::SetW(float value)
{
	data = XMVectorSetW(data, value);
}

float cVector3::GetX()
{
	return XMVectorGetX(data);
}

float cVector3::GetY()
{
	return XMVectorGetY(data);
}

float cVector3::GetZ()
{
	return XMVectorGetZ(data);
}

float cVector3::GetW()
{
	return XMVectorGetW(data);
}

cVector3 cVector3::operator+(const cVector3& value) const
{
	return cVector3(data + value.data);
}

cVector3 cVector3::operator-(const cVector3& value) const
{
	return cVector3(data - value.data);
}

cVector3 cVector3::operator*(const cVector3& value) const
{
	return cVector3(data * value.data);
}

cVector3 cVector3::operator/(const cVector3& value) const
{
	return cVector3(data / value.data);
}

void cVector3::operator+=(const cVector3& value)
{
	data += value.data;
}

void cVector3::operator-=(const cVector3& value)
{
	data -= value.data;
}

void cVector3::operator*=(const cVector3& value)
{
	data *= value.data;
}

void cVector3::operator/=(const cVector3& value)
{
	data /= value.data;
}

cVector3 cVector3::operator+(const float& value) const
{
	return data + XMVectorReplicate(value);
}

cVector3 cVector3::operator-(const float& value) const
{
	return data - XMVectorReplicate(value);
}

cVector3 cVector3::operator*(const float& value) const
{
	return data * XMVectorReplicate(value);
}

cVector3 cVector3::operator/(const float& value) const
{
	return data / XMVectorReplicate(value);
}

void cVector3::operator+=(const float& value)
{
	data += XMVectorReplicate(value);
}

void cVector3::operator-=(const float& value)
{
	data -= XMVectorReplicate(value);
}

void cVector3::operator*=(const float& value)
{
	data *= XMVectorReplicate(value);
}

void cVector3::operator/=(const float& value)
{
	data /= XMVectorReplicate(value);
}

cVector3 operator+(const float value1, const cVector3& value2)
{
	return XMVectorReplicate(value1) + value2.data;
}

cVector3 operator-(const float value1, const cVector3& value2)
{
	return XMVectorReplicate(value1) - value2.data;
}

cVector3 operator*(const float value1, const cVector3& value2)
{
	return XMVectorReplicate(value1) * value2.data;
}

cVector3 operator/(const float value1, const cVector3& value2)
{
	return XMVectorReplicate(value1) / value2.data;
}

cVector3 operator+(const Float3& value1, const cVector3& value2)
{
	return value2 + value1;
}

cVector3 operator-(const Float3& value1, const cVector3& value2)
{
	return value2 - value1;
}

cVector3 operator*(const Float3& value1, const cVector3& value2)
{
	return value2 * value1;
}

cVector3 operator/(const Float3& value1, const cVector3& value2)
{
	return value2 / value1;
}

bool cVector3::operator==(const cVector3& value) const
{
	return XMVector3Equal(data, value.data);
}

float cVector3::operator[](const UINT& index) const
{
	switch (index)
	{
	case 0:
		return XMVectorGetX(data);
	case 1:
		return XMVectorGetY(data);
	case 2:
		return XMVectorGetZ(data);
	default:
		break;
	}

	return 0.0f;
}

float cVector3::Length() const
{
	return XMVectorGetX(XMVector3Length(data));
}

cVector3 cVector3::Normal() const
{
	return XMVector3Normalize(data);
}

void cVector3::Normalize()
{
	data = XMVector3Normalize(data);
}

cVector3 cVector3::Cross(const cVector3& vec1, const cVector3& vec2)
{
	return XMVector3Cross(vec1.data, vec2.data);
}

float cVector3::Dot(const cVector3& vec1, const cVector3& vec2)
{
	return XMVectorGetX(XMVector3Dot(vec1.data, vec2.data));
}
