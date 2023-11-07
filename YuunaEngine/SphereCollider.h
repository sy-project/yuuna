#pragma once

class SphereCollider : public Collider
{
private:
	float radius;

	UINT stackCount;
	UINT sliceCount;

public:
	SphereCollider(float radius = 1.0f, UINT stackCount = 15, UINT sliceCount = 30);
	~SphereCollider();

	virtual bool RayCollision(IN Ray ray, OUT Contact* contact = nullptr) override;
	virtual bool BoxCollision(BoxCollider* collider) override;
	virtual bool SphereCollision(SphereCollider* collider) override;
	virtual bool CapsuleCollision(CapsuleCollider* collider) override;

	float Radius() { return radius * max(GlobalScale().x, max(GlobalScale().y, GlobalScale().z)); }
private:
	virtual void CreateMesh() override;
};