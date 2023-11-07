#include "header.h"

CapsuleCollider::CapsuleCollider(float radius, float height, UINT stackCount, UINT sliceCount)
    : radius(radius), height(height), stackCount(stackCount), sliceCount(sliceCount)
{
    type = CAPSULE;
    CreateMesh();
}

CapsuleCollider::~CapsuleCollider()
{
}

bool CapsuleCollider::RayCollision(IN Ray ray, OUT Contact* contact)
{
	cVector3 direction = Up();

	cVector3 pa = GlobalPos() - direction * Height() * 0.5f;
	cVector3 pb = GlobalPos() + direction * Height() * 0.5f;

	cVector3 ro = ray.position;
	cVector3 rd = ray.direction;

	float r = Radius();

	cVector3 ba = pb - pa;
	cVector3 oa = ro - pa;

	float baba = cVector3::Dot(ba, ba);
	float bard = cVector3::Dot(ba, rd);
	float baoa = cVector3::Dot(ba, oa);
	float rdoa = cVector3::Dot(rd, oa);
	float oaoa = cVector3::Dot(oa, oa);

	float a = baba - bard * bard;
	float b = baba * rdoa - baoa * bard;
	float c = baba * oaoa - baoa * baoa - r * r * baba;
	float h = b * b - a * c;

	if (h >= 0.0f)
	{
		float t = (-b - sqrt(h)) / a;

		float y = baoa + t * bard;

		if (y > 0.0f && y < baba)//Body
		{
			if (contact != nullptr)
			{
				contact->distance = t;
				contact->hitPoint = ray.position + ray.direction * t;
			}			
			return true;
		}

		cVector3 oc = (y <= 0.0f) ? oa : ro - pb;
		b = cVector3::Dot(rd, oc);
		c = cVector3::Dot(oc, oc) - r * r;
		h = b * b - c;
		if (h > 0.0f)
		{
			if (contact != nullptr)
			{
				contact->distance = -b - sqrt(h);
				contact->hitPoint = ray.position + ray.direction * contact->distance;
			}
			return true;
		}
	}

    return false;
}

bool CapsuleCollider::BoxCollision(BoxCollider* collider)
{
	cVector3 direction = Up();
	cVector3 startPos = GlobalPos() - direction * Height() * 0.5f;

	cVector3 A = collider->GlobalPos() - startPos;

	float t = cVector3::Dot(A, direction);
	t = max(0, t);
	t = min(Height(), t);

	cVector3 pointOnLine = startPos + direction * t;	

	return collider->SphereCollision(pointOnLine, Radius());
}

bool CapsuleCollider::SphereCollision(SphereCollider* collider)
{
	cVector3 direction = Up();
	cVector3 startPos = GlobalPos() - direction * Height() * 0.5f;

	cVector3 A = collider->GlobalPos() - startPos;

	float t = cVector3::Dot(A, direction);
	t = max(0, t);
	t = min(Height(), t);

	cVector3 pointOnLine = startPos + direction * t;

	float distance = GameMath::Distance(pointOnLine, collider->GlobalPos());

    return distance <= (Radius() + collider->Radius());
}

bool CapsuleCollider::CapsuleCollision(CapsuleCollider* collider)
{
	cVector3 aDirection = Up();

	cVector3 aA = GlobalPos() - aDirection * Height() * 0.5f;
	cVector3 aB = GlobalPos() + aDirection * Height() * 0.5f;

	cVector3 bDirection = collider->Up();

	cVector3 bA = collider->GlobalPos() - bDirection * collider->Height() * 0.5f;
	cVector3 bB = collider->GlobalPos() + bDirection * collider->Height() * 0.5f;

	cVector3 v0 = bA - aA;
	cVector3 v1 = bB - aA;
	cVector3 v2 = bA - aB;
	cVector3 v3 = bB - aB;

	float d0 = cVector3::Dot(v0, v0);
	float d1 = cVector3::Dot(v1, v1);
	float d2 = cVector3::Dot(v2, v2);
	float d3 = cVector3::Dot(v3, v3);

	cVector3 bestA;
	if (d2 < d0 || d2 < d1 || d3 < d0 || d3 > d1)
		bestA = aB;
	else
		bestA = aA;

	cVector3 bestB = GameMath::ClosestPointOnLineSegment(bA, bB, bestA);
	bestA = GameMath::ClosestPointOnLineSegment(aA, aB, bestB);

	float distance = GameMath::Distance(bestA, bestB);

    return distance <= (Radius() + collider->Radius());
}

void CapsuleCollider::CreateMesh()
{
	float phiStep = XM_PI / stackCount;
	float thetaStep = XM_2PI / sliceCount;

	for (UINT i = 0; i <= stackCount; i++)
	{
		float phi = i * phiStep;

		for (UINT j = 0; j <= sliceCount; j++)
		{
			float theta = j * thetaStep;

			Vertex vertex;

			vertex.position.x = sin(phi) * cos(theta) * radius;
			vertex.position.y = cos(phi) * radius;
			vertex.position.z = sin(phi) * sin(theta) * radius;

			if (vertex.position.y > 0)
				vertex.position.y += height * 0.5f;
			else
				vertex.position.y -= height * 0.5f;

			vertices.emplace_back(vertex);
		}
	}

	for (UINT i = 0; i < stackCount; i++)
	{
		for (UINT j = 0; j < sliceCount; j++)
		{
			indices.emplace_back((sliceCount + 1) * i + j);//0
			indices.emplace_back((sliceCount + 1) * i + j + 1);//1			

			indices.emplace_back((sliceCount + 1) * i + j);//0
			indices.emplace_back((sliceCount + 1) * (i + 1) + j);//2
		}
	}

	mesh = new Mesh(vertices.data(), sizeof(Vertex), vertices.size(),
		indices.data(), indices.size());
}
