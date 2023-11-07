#include "header.h"

bool Transform::isAxisDraw = true;

Transform::Transform(string tag)
	: tag(tag), 
	position(0, 0, 0), rotation(0, 0, 0), scale(1, 1, 1),
	globalPosition(0, 0, 0), globalRotation(0, 0, 0), globalScale(1, 1, 1),
	pivot(0, 0, 0), parent(nullptr), isActive(true)
{
	world = XMMatrixIdentity();
	worldBuffer = new MatrixBuffer();

	material = new Material(L"Transform");
	CreateAxis();
	transformBuffer = new MatrixBuffer();
}

Transform::~Transform()
{
	delete worldBuffer;	
	delete transformBuffer;
	delete material;
}

void Transform::UpdateWorld()
{
	world = XMMatrixTransformation(pivot.data, XMQuaternionIdentity(),
		scale.data, pivot.data,
		XMQuaternionRotationRollPitchYaw(rotation.x, rotation.y,
			rotation.z), position.data);

	if (parent != nullptr)
		world *= *parent;

	XMMatrixDecompose(&globalScale.data, &globalRotation.data,
		&globalPosition.data, world);

	worldBuffer->Set(world);
}

void Transform::RenderAxis()
{
	if (!isAxisDraw)
		return;

	cVector3 scale(1, 1, 1);
	Matrix matrix = XMMatrixTransformation(pivot.data, XMQuaternionIdentity(),
		scale.data, pivot.data, globalRotation.data, globalPosition.data);

	transformBuffer->Set(matrix);
	transformBuffer->SetVSBuffer(0);

	mesh->IASet();
	material->Set();	

	Device::Get()->GetDeviceContext()->DrawIndexed(indices.size(), 0, 0);
}

void Transform::SetWorldBuffer(UINT slot)
{
	UpdateWorld();
	worldBuffer->SetVSBuffer(slot);
}

cVector3 Transform::Forward()
{
	return XMVector3Normalize(XMVector3TransformNormal(kForward, world));
}

cVector3 Transform::Up()
{
	return XMVector3Normalize(XMVector3TransformNormal(kUp, world));
}

cVector3 Transform::Right()
{
	return XMVector3Normalize(XMVector3TransformNormal(kRight, world));
}

void Transform::CreateAxis()
{
	float length = 3.0f;
	float thickness = 0.5f;

	//Axis X		
	Float4 color = { 1, 0, 0, 1 };
	vertices.emplace_back(Float3(0, 0, 0), color);
	vertices.emplace_back(Float3(length, 0, 0), color);
	vertices.emplace_back(Float3(length, thickness, 0), color);
	vertices.emplace_back(Float3(length, 0, thickness), color);

	indices.emplace_back(0);
	indices.emplace_back(2);
	indices.emplace_back(1);

	indices.emplace_back(0);
	indices.emplace_back(1);
	indices.emplace_back(3);

	indices.emplace_back(0);
	indices.emplace_back(3);
	indices.emplace_back(2);

	indices.emplace_back(1);
	indices.emplace_back(2);
	indices.emplace_back(3);

	//Axis Y
	color = { 0, 1, 0, 1 };
	vertices.emplace_back(Float3(0, 0, 0), color);
	vertices.emplace_back(Float3(0, length, 0), color);
	vertices.emplace_back(Float3(0, length, thickness), color);
	vertices.emplace_back(Float3(thickness, length, 0), color);

	indices.emplace_back(4);
	indices.emplace_back(6);
	indices.emplace_back(5);

	indices.emplace_back(4);
	indices.emplace_back(5);
	indices.emplace_back(7);

	indices.emplace_back(4);
	indices.emplace_back(7);
	indices.emplace_back(6);

	indices.emplace_back(5);
	indices.emplace_back(6);
	indices.emplace_back(7);

	//Axis Z
	color = { 0, 0, 1, 1 };
	vertices.emplace_back(Float3(0, 0, 0), color);
	vertices.emplace_back(Float3(0, 0, length), color);
	vertices.emplace_back(Float3(thickness, 0, length), color);
	vertices.emplace_back(Float3(0, thickness, length), color);

	indices.emplace_back(8);
	indices.emplace_back(10);
	indices.emplace_back(9);

	indices.emplace_back(8);
	indices.emplace_back(9);
	indices.emplace_back(11);

	indices.emplace_back(8);
	indices.emplace_back(11);
	indices.emplace_back(10);

	indices.emplace_back(9);
	indices.emplace_back(10);
	indices.emplace_back(11);

	mesh = new Mesh(vertices.data(), sizeof(VertexColor), vertices.size(),
		indices.data(), indices.size());
}
