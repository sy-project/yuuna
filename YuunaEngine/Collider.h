#pragma once

struct Ray
{
	cVector3 position;
	cVector3 direction;
};

struct Contact
{
	cVector3 hitPoint;
	float distance;
};

class BoxCollider;
class SphereCollider;
class CapsuleCollider;

class Collider : public Transform
{
public:
	enum Type
	{
		BOX,
		SPHERE,
		CAPSULE,
		NONE
	};
protected:
	Material* material;
	Mesh* mesh;

	Type type;

	vector<Vertex> vertices;
	vector<UINT> indices;

public:
	Collider();
	virtual ~Collider();

	bool Collision(Collider* collider);
	virtual bool RayCollision(IN Ray ray, OUT Contact* contact = nullptr) = 0;
	virtual bool BoxCollision(BoxCollider* collider) = 0;
	virtual bool SphereCollision(SphereCollider* collider) = 0;
	virtual bool CapsuleCollision(CapsuleCollider* collider) = 0;

	virtual void CreateMesh() = 0;

	void Render();

	void SetColor(Float4 color) { material->GetBuffer()->data.diffuse = color; }
};