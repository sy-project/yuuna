#include "header.h"

BoxCollider::BoxCollider(cVector3 minBox, cVector3 maxBox)
    : minBox(minBox), maxBox(maxBox)
{
    type = BOX;
    CreateMesh();
}

BoxCollider::~BoxCollider()
{
}

bool BoxCollider::RayCollision(IN Ray ray, OUT Contact* contact)
{
    if (ray.direction.Length() == 0.0f)
        return false;

    Contact temp;
    temp.distance = FLT_MAX;

    UINT face[] = {
        0, 1, 2, 3,//F
        4, 5, 6, 7,//B
        0, 1, 5, 4,//L
        1, 5, 6, 2,//R
        2, 3, 7, 6,//U
        0, 3, 7, 4,//D
    };

    for (UINT i = 0; i < 6; i++)
    {
        cVector3 p[4];

        p[0] = vertices[face[i * 4 + 0]].position;
        p[1] = vertices[face[i * 4 + 1]].position;
        p[2] = vertices[face[i * 4 + 2]].position;
        p[3] = vertices[face[i * 4 + 3]].position;

        p[0] = XMVector3TransformCoord(p[0].data, world);
        p[1] = XMVector3TransformCoord(p[1].data, world);
        p[2] = XMVector3TransformCoord(p[2].data, world);
        p[3] = XMVector3TransformCoord(p[3].data, world);

        float dist;
        if (Intersects(ray.position.data, ray.direction.data,
            p[0].data, p[1].data, p[2].data, dist))
        {
            if (dist < temp.distance)
            {
                temp.distance = dist;
                temp.hitPoint = ray.position + ray.direction * dist;
            }
        }

        if (Intersects(ray.position.data, ray.direction.data,
            p[0].data, p[3].data, p[2].data, dist))
        {
            if (dist < temp.distance)
            {
                temp.distance = dist;
                temp.hitPoint = ray.position + ray.direction * dist;
            }
        }
    }

    if (temp.distance == FLT_MAX)
        return false;

    if (contact != nullptr)
    {
        contact->distance = temp.distance;
        contact->hitPoint = temp.hitPoint;
    }

    return true;
}
/*
bool BoxCollider::RayCollision(IN Ray ray, OUT Contact* contact)
{
    Vector3 min = minBox;
    Vector3 max = maxBox;

    Matrix invWorld = XMMatrixInverse(nullptr, world);

    Ray r;
    r.position = XMVector3TransformCoord(ray.position.data, invWorld);
    r.direction = XMVector3TransformNormal(ray.direction.data, invWorld);
    r.direction.Normalize();

    float tmin = (min.x - r.position.x) / r.direction.x;
    float tmax = (max.x - r.position.x) / r.direction.x;

    if (tmin > tmax) swap(tmin, tmax);

    float tymin = (min.y - r.position.y) / r.direction.y;
    float tymax = (max.y - r.position.y) / r.direction.y;

    if (tymin > tymax) swap(tymin, tymax);

    if ((tmin > tymax) || (tymin > tmax))
        return false;

    if (tymin > tmin)
        tmin = tymin;

    if (tymax < tmax)
        tmax = tymax;

    float tzmin = (min.z - r.position.z) / r.direction.z;
    float tzmax = (max.z - r.position.z) / r.direction.z;

    if (tzmin > tzmax) swap(tzmin, tzmax);

    if ((tmin > tzmax) || (tzmin > tmax))
        return false;    

    return true;
}*/

bool BoxCollider::BoxCollision(BoxCollider* collider)
{
    Obb box1 = GetObb();
    Obb box2 = collider->GetObb();

    cVector3 D = box2.position - box1.position;

    for(UINT i = 0; i < 3; i++)
    {
        if (SeperateAxis(D, box1.axis[i], box1, box2)) return false;
        if (SeperateAxis(D, box2.axis[i], box1, box2)) return false;
    }

    for (UINT i = 0; i < 3; i++)
    {
        for (UINT j = 0; j < 3; j++)
        {
            if (box1.axis[i] == box2.axis[j]) return true;
        }
    }

    for (UINT i = 0; i < 3; i++)
    {
        for (UINT j = 0; j < 3; j++)
        {
            cVector3 cross = cVector3::Cross(box1.axis[i], box2.axis[j]);
            if (SeperateAxis(D, cross, box1, box2)) return false;
        }
    }

    return true;
}

bool BoxCollider::SphereCollision(SphereCollider* collider)
{
    Matrix T = XMMatrixTranslation(GlobalPos().x, GlobalPos().y, GlobalPos().z);
    Matrix R = XMMatrixRotationQuaternion(GlobalRot().data);

    Matrix invWorld = XMMatrixInverse(nullptr, R * T);

    cVector3 spherePos = XMVector3TransformCoord(collider->GlobalPos().data, invWorld);

    cVector3 tempMin = minBox * GlobalScale();
    cVector3 tempMax = maxBox * GlobalScale();

    cVector3 temp;
    temp.x = max(tempMin.x, min(spherePos.x, tempMax.x));
    temp.y = max(tempMin.y, min(spherePos.y, tempMax.y));
    temp.z = max(tempMin.z, min(spherePos.z, tempMax.z));

    temp -= spherePos;

    return temp.Length() <= collider->Radius();
}

/*
bool BoxCollider::SphereCollision(SphereCollider* collider)
{
    Obb box = GetObb();

    Vector3 pos = box.position;

    for (UINT i = 0; i < 3; i++)
    {
        float length = Vector3::Dot(box.axis[i], collider->GlobalPos() - box.position);

        float mult = (length < 0.0f) ? -1.0f : 1.0f;

        length = min(abs(length), box.halfSize[i]);
        pos += box.axis[i] * length * mult;
    }

    float distance = (collider->GlobalPos() - pos).Length();

    return distance <= collider->Radius();
}*/

bool BoxCollider::CapsuleCollision(CapsuleCollider* collider)
{
    return collider->BoxCollision(this);
}

bool BoxCollider::SphereCollision(cVector3 center, float radius)
{
    Matrix T = XMMatrixTranslation(GlobalPos().x, GlobalPos().y, GlobalPos().z);
    Matrix R = XMMatrixRotationQuaternion(GlobalRot().data);

    Matrix invWorld = XMMatrixInverse(nullptr, R * T);

    cVector3 spherePos = XMVector3TransformCoord(center.data, invWorld);

    cVector3 tempMin = minBox * GlobalScale();
    cVector3 tempMax = maxBox * GlobalScale();

    cVector3 temp;
    temp.x = max(tempMin.x, min(spherePos.x, tempMax.x));
    temp.y = max(tempMin.y, min(spherePos.y, tempMax.y));
    temp.z = max(tempMin.z, min(spherePos.z, tempMax.z));

    temp -= spherePos;

    return temp.Length() <= radius;
}

cVector3 BoxCollider::MinBox()
{
    return XMVector3TransformCoord(minBox.data, world);
}

cVector3 BoxCollider::MaxBox()
{
    return XMVector3TransformCoord(maxBox.data, world);
}

Obb BoxCollider::GetObb()
{
    obb.position = GlobalPos();

    obb.axis[0] = Right();
    obb.axis[1] = Up();
    obb.axis[2] = Forward();

    obb.halfSize = (minBox - maxBox) * 0.5f * GlobalScale();

    return obb;
}

void BoxCollider::CreateMesh()
{
    vertices.emplace_back(minBox.x, minBox.y, minBox.z);
    vertices.emplace_back(minBox.x, maxBox.y, minBox.z);
    vertices.emplace_back(maxBox.x, maxBox.y, minBox.z);
    vertices.emplace_back(maxBox.x, minBox.y, minBox.z);

    vertices.emplace_back(minBox.x, minBox.y, maxBox.z);
    vertices.emplace_back(minBox.x, maxBox.y, maxBox.z);
    vertices.emplace_back(maxBox.x, maxBox.y, maxBox.z);
    vertices.emplace_back(maxBox.x, minBox.y, maxBox.z);

    indices = {
        0, 1, 1, 2, 2, 3, 3, 0,
        4, 5, 5, 6, 6, 7, 7, 4,
        0, 4, 1, 5, 2, 6, 3, 7
    };

    mesh = new Mesh(vertices.data(), sizeof(Vertex), vertices.size(),
        indices.data(), indices.size());
}

bool BoxCollider::SeperateAxis(cVector3 D, cVector3 axis, Obb box1, Obb box2)
{
    float distance = abs(cVector3::Dot(D, axis));

    float a = 0.0f;
    float b = 0.0f;

    for (UINT i = 0; i < 3; i++)
    {
        a += abs(cVector3::Dot(box1.axis[i] * box1.halfSize[i], axis));
        b += abs(cVector3::Dot(box2.axis[i] * box2.halfSize[i], axis));
    }

    return distance > a + b;
}
