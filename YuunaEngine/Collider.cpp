#include "header.h"

Collider::Collider()
{
    material = new Material(L"Collider");
    material->GetBuffer()->data.diffuse = Float4(0, 1, 0, 1);
}

Collider::~Collider()
{
    delete material;
    delete mesh;
}

bool Collider::Collision(Collider* collider)
{    
    switch (collider->type)
    {
    case Collider::BOX:
        return BoxCollision(static_cast<BoxCollider*>(collider));
        break;
    case Collider::SPHERE:
        return SphereCollision(static_cast<SphereCollider*>(collider));
        break;
    case Collider::CAPSULE:
        return CapsuleCollision(static_cast<CapsuleCollider*>(collider));
        break;    
    }

    return false;
}

void Collider::Render()
{
    SetWorldBuffer();

    mesh->IASet(D3D_PRIMITIVE_TOPOLOGY_LINELIST);
    material->Set();

    Device::Get()->GetDeviceContext()->DrawIndexed(indices.size(), 0, 0);
}
