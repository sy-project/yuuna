#include "header.h"
#include "Math.h"

float GameMath::Saturate(const float& value)
{
    return max(0.0f, min(1.0f, value));
}

int GameMath::Random(int min, int max)
{
    return rand() % (max - min) + min;
}

float GameMath::Random(float min, int max)
{
    float normal = rand() / (float)RAND_MAX;
    return min + (max - min) * normal;
}

float GameMath::Distance(const cVector3& v1, const cVector3& v2)
{
    return (v1 - v2).Length();
}

cVector3 GameMath::ClosestPointOnLineSegment(const cVector3& v1, const cVector3& v2, const cVector3& point)
{
    cVector3 line = v2 - v1;
    float t = cVector3::Dot(line, point - v1) / cVector3::Dot(line, line);
    t = Saturate(t);

    return v1 + t * line;
}

cVector3 GameMath::WorldToScreen(const cVector3& worldPos)
{
    cVector3 screenPos;

    screenPos = XMVector3TransformCoord(worldPos.data, Environment::Get()->GetMainCamera()->GetView());
    screenPos = XMVector3TransformCoord(screenPos.data, Environment::Get()->GetProjection());
    //NDC°ø°£ ÁÂÇ¥(-1 ~ 1) -> È­¸éÁÂÇ¥(0 ~ WIN_WIDTH)

    screenPos.y *= -1;

    screenPos = (screenPos + 1.0f) * 0.5f;

    screenPos.x *= WINDOW_WIDTH;
    screenPos.y *= WINDOW_HEIGHT;

    return screenPos;
}
