#pragma once

namespace GameMath
{
	float Saturate(const float& value);

	int Random(int min, int max);
	float Random(float min, int max);

	float Distance(const cVector3& v1, const cVector3& v2);
	cVector3 ClosestPointOnLineSegment(const cVector3& v1, const cVector3& v2, const cVector3& point);

	cVector3 WorldToScreen(const cVector3& worldPos);
}