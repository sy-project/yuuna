#pragma once

struct Obb
{
	cVector3 position;

	cVector3 axis[3];
	cVector3 halfSize;
};

class BoxCollider : public Collider
{
private:
	cVector3 minBox;
	cVector3 maxBox;

	Obb obb;
public:
	BoxCollider(cVector3 minBox = cVector3(-0.5f, -0.5f, -0.5f),
		cVector3 maxBox = cVector3(0.5f, 0.5f, 0.5f));
	~BoxCollider();

	virtual bool RayCollision(IN Ray ray, OUT Contact* contact = nullptr) override;
	virtual bool BoxCollision(BoxCollider* collider) override;
	virtual bool SphereCollision(SphereCollider* collider) override;
	virtual bool CapsuleCollision(CapsuleCollider* collider) override;

	bool SphereCollision(cVector3 center, float radius);

	cVector3 MinBox();
	cVector3 MaxBox();

	Obb GetObb();
private:
	virtual void CreateMesh() override;

	bool SeperateAxis(cVector3 D, cVector3 axis, Obb box1, Obb box2);
};