#pragma once

class Camera : public Transform
{
private:
	class ViewBuffer : public ConstBuffer
	{
	private:
		struct Data
		{
			Matrix matrix;
			Matrix invMatrix;
		}data;

	public:
		ViewBuffer() : ConstBuffer(&data, sizeof(Data))
		{
			data.matrix = XMMatrixIdentity();
			data.invMatrix = XMMatrixIdentity();
		}

		void Set(Matrix value)
		{
			data.matrix = XMMatrixTranspose(value);
			Matrix temp = XMMatrixInverse(nullptr, value);
			data.invMatrix = XMMatrixTranspose(temp);
		}
	};

	float moveSpeed;
	float rotSpeed;

	ViewBuffer* viewBuffer;
	Matrix view;

	cVector3 oldPos;

	float distance;
	float height;

	cVector3 offset;

	cVector3 destPos;
	float destRot;

	float moveDamping;
	float rotDamping;

	float rotY;	

	Matrix rotMatrix;

	Transform* target;
public:
	Camera();
	~Camera();

	void Update();

	void FreeMode();
	void FollowMode();

	void FreeMove();
	void FollowMove();

	void FollowControl();

	void Rotation();
	void View();

	void PostRender();

	void SetVS(UINT slot = 1);

	Ray ScreenPointToRay(cVector3 pos);

	void SetTarget(Transform* value) { target = value; }
	Matrix GetView() { return view; }
	ViewBuffer* GetViewBuffer() { return viewBuffer; }
};