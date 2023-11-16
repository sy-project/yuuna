#include "header.h"

Camera::Camera()
	: moveSpeed(20.0f), rotSpeed(5.0f), distance(60), height(60),
	offset(0, 5, 0), moveDamping(5), rotDamping(0), destRot(0),
	rotY(0), target(nullptr)
{
	viewBuffer = new ViewBuffer();
	oldPos = cControl::Get()->GetMouse();
}

Camera::~Camera()
{
	delete viewBuffer;
}

void Camera::Update()
{
	if (target != nullptr)
	{
		FollowMode();
	}
	else
	{
		FreeMode();
	}		
}

void Camera::FreeMode()
{
	FreeMove();
	Rotation();
	View();
}

void Camera::FollowMode()
{
	FollowMove();
}

void Camera::FreeMove()
{
	if (cControl::Get()->Press(VK_RBUTTON))
	{
		if (cControl::Get()->Press('W'))
			position += Forward() * moveSpeed * cTimer::Get()->GetElapsedTime();
		if (cControl::Get()->Press('S'))
			position -= Forward() * moveSpeed * cTimer::Get()->GetElapsedTime();
		if (cControl::Get()->Press('A'))
			position -= Right() * moveSpeed * cTimer::Get()->GetElapsedTime();
		if (cControl::Get()->Press('D'))
			position += Right() * moveSpeed * cTimer::Get()->GetElapsedTime();
		if (cControl::Get()->Press('Q'))
			position -= Up() * moveSpeed * cTimer::Get()->GetElapsedTime();
		if (cControl::Get()->Press('E'))
			position += Up() * moveSpeed * cTimer::Get()->GetElapsedTime();
	}

	position += Forward() * cControl::Get()->GetWheel() * moveSpeed * cTimer::Get()->GetElapsedTime();
}

void Camera::FollowMove()
{
	if (rotDamping > 0.0f)
	{
		if (target->rotation.y != destRot)
		{
			destRot = LERP(destRot, target->rotation.y + XM_PI, rotDamping * cTimer::Get()->GetElapsedTime());
		}

		rotMatrix = XMMatrixRotationY(destRot);
	}
	else
	{		
		FollowControl();
		rotMatrix = XMMatrixRotationY(rotY);
	}

	cVector3 forward = XMVector3TransformNormal(kForward, rotMatrix);
	destPos = forward * -distance;

	destPos += target->GlobalPos();
	destPos.y += height;

	position = LERP(position, destPos, moveDamping * cTimer::Get()->GetElapsedTime());

	cVector3 tempOffset = XMVector3TransformCoord(offset.data, rotMatrix);

	view = XMMatrixLookAtLH(position.data, (target->GlobalPos() + tempOffset).data,
		Up().data);
	viewBuffer->Set(view);
}

void Camera::FollowControl()
{
	if (cControl::Get()->Press(VK_RBUTTON))
	{
		cVector3 value = cControl::Get()->GetMouse() - oldPos;

		rotY += value.x * rotSpeed * cTimer::Get()->GetElapsedTime();
	}

	oldPos = cControl::Get()->GetMouse();
}

void Camera::Rotation()
{
	if (cControl::Get()->Press(VK_RBUTTON))
	{
		cVector3 value = cControl::Get()->GetMouse() - oldPos;

		rotation.x += value.y * rotSpeed * cTimer::Get()->GetElapsedTime();
		rotation.y += value.x * rotSpeed * cTimer::Get()->GetElapsedTime();
	}	

	oldPos = cControl::Get()->GetMouse();
}

void Camera::View()
{
	UpdateWorld();

	cVector3 focus = position + Forward();
	view = XMMatrixLookAtLH(position.data, focus.data, Up().data);

	viewBuffer->Set(view);
}

void Camera::PostRender()
{
	ImGuiManager::Get()->OpenImGuiWindow("Camera");
	ImGui::Text("CameraInfo");
	ImGui::Text("CamPos : %.1f, %.1f, %.1f", position.x, position.y, position.z);
	ImGui::Text("CamRot : %.1f, %.1f, %.1f", rotation.x, rotation.y, rotation.z);
	if (target == nullptr)
	{
		ImGui::SliderFloat("MoveSpeed", &moveSpeed, 0, 100);
		ImGui::SliderFloat("RotSpeed", &rotSpeed, 0, 10);
	}
	else
	{
		ImGui::SliderFloat("CamDistance", &distance, -10.0f, 100.0f);
		ImGui::SliderFloat("CamHeight", &height, -10.0f, 100.0f);
		ImGui::SliderFloat("CamMoveDamping", &moveDamping, 0.0f, 30.0f);
		ImGui::SliderFloat("CamRotDamping", &rotDamping, 0.0f, 30.0f);
		ImGui::SliderFloat3("CamOffset", (float*)&offset, -20.0f, 20.0f);
	}
	ImGui::Spacing();
	ImGui::End();
}

void Camera::SetVS(UINT slot)
{
	viewBuffer->SetVSBuffer(slot);
}
Ray Camera::ScreenPointToRay(cVector3 pos)
{
	Float2 screenSize(WINDOW_WIDTH, WINDOW_HEIGHT);

	Float2 point;
	point.x = ((2 * pos.x) / screenSize.x) - 1.0f;
	point.y = (((2 * pos.y) / screenSize.y) - 1.0f) * -1.0f;

	Matrix projection = Environment::Get()->GetProjection();

	Float4x4 temp;
	XMStoreFloat4x4(&temp, projection);

	point.x /= temp._11;
	point.y /= temp._22;

	Ray ray;
	ray.position = position;

	Matrix invView = XMMatrixInverse(nullptr, view);

	cVector3 tempPos(point.x, point.y, 1.0f);

	ray.direction = XMVector3TransformNormal(tempPos.data, invView);
	ray.direction.Normalize();

	return ray;
}
