#include "cuda_math.cuh"
#include <math.h>
#include "cuda_profiler_api.h"

__device__ bool isInside2DTriangle(Vector::Vector2D p, Vector::Vector2D a, Vector::Vector2D b, Vector::Vector2D c) {
    Vector::Vector2D AB = { b.x - a.x, b.y - a.y };
    Vector::Vector2D BC = { c.x - b.x, c.y - b.y };
    Vector::Vector2D CA = { a.x - c.x, a.y - c.y };

    Vector::Vector2D AP = { p.x - a.x, p.y - a.y };
    Vector::Vector2D BP = { p.x - b.x, p.y - b.y };
    Vector::Vector2D CP = { p.x - c.x, p.y - c.y };

    float cross1 = AB.x * AP.y - AB.y * AP.x;  // AB x AP
    float cross2 = BC.x * BP.y - BC.y * BP.x;  // BC x BP
    float cross3 = CA.x * CP.y - CA.y * CP.x;  // CA x CP

    return (cross1 >= 0 && cross2 >= 0 && cross3 >= 0) || (cross1 <= 0 && cross2 <= 0 && cross3 <= 0);
}
__device__ bool isInside3DTriangle(const Vector::Vector3D& p, const Vector::Vector3D& a, const Vector::Vector3D& b, const Vector::Vector3D& c)
{
    float area = (b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y);
    float w0 = (p.x - b.x) * (p.y - c.y) - (p.y - b.y) * (p.x - c.x);
    float w1 = (p.x - c.x) * (p.y - a.y) - (p.y - c.y) * (p.x - a.x);
    float w2 = (p.x - a.x) * (p.y - b.y) - (p.y - a.y) * (p.x - b.x);

    // 삼각형의 방향 일치 여부 검사
    return (area >= 0 && w0 >= 0 && w1 >= 0 && w2 >= 0) ||
        (area <= 0 && w0 <= 0 && w1 <= 0 && w2 <= 0);
}
__global__ void GMath_Test(int* c, const int* a, const int* b)
{
    int i = threadIdx.x;
    //GMultiply((float)c[i], (float)a[i], (float)b[i]);
    //if(i % 2 == 0)
        c[i] = a[i] + b[i];
    //else
    //    c[i] = b[i] - a[i];
}
__device__ void GInverse(float* out, const float* in, int _index)
{
    out[_index] = (1.0f / in[_index]);
}
__device__ void GFMultiply(float* out, const float* x, const float* y, int _index)
{
    out[_index] = x[_index] * y[_index];
}
__device__ void GIMultiply(int* out, const int* x, const int* y, int _index)
{
    out[_index] = x[_index] * y[_index];
}
__device__ void GNormalize(float* out, const float* InVec[3], const float* OutVec[3], int _index)
{
    float* Length, InvLength;
    cudaMalloc((void**)&Length, sizeof(out) * sizeof(float));
    cudaMalloc((void**)&InvLength, sizeof(out) * sizeof(float));

    float* mul0[3];
    cudaMalloc((void**)&mul0[0], sizeof(out) * sizeof(float));
    cudaMalloc((void**)&mul0[1], sizeof(out) * sizeof(float));
    cudaMalloc((void**)&mul0[2], sizeof(out) * sizeof(float));

    GFMultiply(mul0[0], InVec[0], InVec[0], _index);
    GFMultiply(mul0[1], InVec[1], InVec[1], _index);
    GFMultiply(mul0[2], InVec[2], InVec[2], _index);

    Length[_index] = mul0[0][_index] + mul0[1][_index] + mul0[2][_index];
}
__device__ void testfunc(double* val)
{
    int i = threadIdx.x;
    sqrt(val[i]);
}
__host__ __device__ Vector::Vector3D normalize(Vector::Vector3D v)
{
    float length = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    if (length == 0.0f)
        return { 0.0f, 0.0f, 0.0f };

    return {
        v.x / length,
        v.y / length,
        v.z / length
    };
}
__host__ __device__ float dot(Vector::Vector3D a, Vector::Vector3D b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
__host__ Matrix4x4 Perspective(float fovDeg, float aspect, float _near, float _far)
{
    Matrix4x4 mat = {};
    float fovRad = fovDeg * 3.1415926535f / 180.0f;
    float f = 1.0f / tanf(fovRad / 2.0f);

    mat.m[0][0] = f / aspect;
    mat.m[1][1] = f;
    mat.m[2][2] = (_far + _near) / (_near - _far);
    mat.m[2][3] = -1.0f;
    mat.m[3][2] = (2.0f * _far * _near) / (_near - _far);

    return mat;
}
__host__ Matrix4x4 PerspectiveNoClip(float fovDeg, float aspect)
{
    Matrix4x4 mat = {};

    float fovRad = fovDeg * 3.1415926535f / 180.0f;
    float f = 1.0f / tanf(fovRad / 2.0f);

    mat.m[0][0] = f / aspect;
    mat.m[1][1] = f;
    mat.m[2][2] = 1.0f;
    mat.m[2][3] = -1.0f;
    mat.m[3][2] = 0.0f;
    mat.m[3][3] = 0.0f;

    return mat;
}
__host__ Matrix4x4 Ortho(float left, float right, float bottom, float top, float _near, float _far)
{
    Matrix4x4 mat = {};

    mat.m[0][0] = 2.0f / (right - left);
    mat.m[1][1] = 2.0f / (top - bottom);
    mat.m[2][2] = -2.0f / (_far - _near);
    mat.m[3][0] = -(right + left) / (right - left);
    mat.m[3][1] = -(top + bottom) / (top - bottom);
    mat.m[3][2] = -(_far + _near) / (_far - _near);
    mat.m[3][3] = 1.0f;

    return mat;
}
__device__ Vector::Vector3D mulMat_Vec(const Matrix4x4& mat, const Vector::Vector3D& v)
{
    float x = v.x * mat.m[0][0] + v.y * mat.m[1][0] + v.z * mat.m[2][0] + mat.m[3][0];
    float y = v.x * mat.m[0][1] + v.y * mat.m[1][1] + v.z * mat.m[2][1] + mat.m[3][1];
    float z = v.x * mat.m[0][2] + v.y * mat.m[1][2] + v.z * mat.m[2][2] + mat.m[3][2];
    float w = v.x * mat.m[0][3] + v.y * mat.m[1][3] + v.z * mat.m[2][3] + mat.m[3][3];

    if (w != 0.0f) {
        x /= w;
        y /= w;
        z /= w;
    }

    return { x, y, z };
}
__host__ __device__ Matrix4x4 mulMat(const Matrix4x4& mat1, const Matrix4x4& mat2)
{
    Matrix4x4 result = {};

    for (int row = 0; row < 4; ++row)
    {
        for (int col = 0; col < 4; ++col)
        {
            result.m[row][col] =
                mat1.m[row][0] * mat2.m[0][col] +
                mat1.m[row][1] * mat2.m[1][col] +
                mat1.m[row][2] * mat2.m[2][col] +
                mat1.m[row][3] * mat2.m[3][col];
        }
    }

    return result;
}
__device__ Vector::Vector3D NDCtoScreen(const Vector::Vector3D& ndc, int screenWidth, int screenHeight)
{
    float screenX = (ndc.x + 1.0f) * 0.5f * screenWidth;
    float screenY = (1.0f - ndc.y) * 0.5f * screenHeight;  // y축 반전
    return { screenX, screenY, ndc.z };
}
__host__ __device__ Vector::Vector3D cross(const Vector::Vector3D& a, const Vector::Vector3D& b)
{
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}
__device__ Vector::Vector3D ProjectVertex(const Vector::Vector3D& v, const Matrix4x4& viewProj, int screenWidth, int screenHeight)
{
    Vector::Vector3D clip = mulMat_Vec(viewProj, v);
    float sx = (clip.x + 1.0f) * 0.5f * screenWidth;
    float sy = (1.0f - clip.y) * 0.5f * screenHeight;
    return { sx, sy, clip.z };
}
__host__ __device__ Matrix4x4 LookAt(const Vector::Vector3D& eye, const Vector::Vector3D& target, const Vector::Vector3D& up)
{
    Vector::Vector3D z = normalize({ target.x - eye.x, target.y - eye.y, target.z - eye.z });
    Vector::Vector3D x = normalize(cross(up, z));
    Vector::Vector3D y = cross(z, x);

    Matrix4x4 mat = Matrix4x4::identity();
    mat.m[0][0] = x.x; mat.m[1][0] = x.y; mat.m[2][0] = x.z;
    mat.m[0][1] = y.x; mat.m[1][1] = y.y; mat.m[2][1] = y.z;
    mat.m[0][2] = z.x; mat.m[1][2] = z.y; mat.m[2][2] = z.z;

    mat.m[3][0] = -dot(x, eye);
    mat.m[3][1] = -dot(y, eye);
    mat.m[3][2] = -dot(z, eye);
    return mat;
}
__global__ void GMath_Test1(int* c, const int* a, const int* b) 
{
    int i = threadIdx.x;
    //c[i] = a[i] * b[i];
    GIMultiply(c, a, b, i);
    //double* db = new double[8];
    //testfunc(db);
    //GMath_Test KERNEL_ARG2(1,1) (c, a, b);
}
