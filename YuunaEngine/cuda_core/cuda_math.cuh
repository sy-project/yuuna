#pragma once
#ifndef CUDACC
#define CUDACC
#endif
#include <cmath>
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>
#include "cuda_header.h"
#include <Types.h>
extern __device__ bool isInside2DTriangle(Vector::Vector2D p, Vector::Vector2D a, Vector::Vector2D b, Vector::Vector2D c);
extern __device__ bool isInside3DTriangle(const Vector::Vector3D& p, const Vector::Vector3D& a, const Vector::Vector3D& b, const Vector::Vector3D& c);
extern __device__ void GInverse(float* out, const float* in, int _index);
extern __device__ void GFMultiply(float* out, const float* x, const float* y, int _index);
extern __device__ void GIMultiply(int* out, const int* x, const int* y, int _index);
extern __device__ void GNormalize(float* out, const float* InVec[3], const float* OutVec[3], int _index);
extern __device__ void testfunc(double* val);
extern __host__ __device__ Vector::Vector3D normalize(Vector::Vector3D v);
extern __host__ __device__ float dot(Vector::Vector3D a, Vector::Vector3D b);
extern __host__ Matrix4x4 Perspective(float fovDeg, float aspect, float _near, float _far);
extern __host__ Matrix4x4 PerspectiveNoClip(float fovDeg, float aspect);
extern __host__ Matrix4x4 Ortho(float left, float right, float bottom, float top, float _near, float _far);
extern __device__ Vector::Vector3D mulMat_Vec(const Matrix4x4& mat, const Vector::Vector3D& v);
extern __host__ __device__ Matrix4x4 mulMat(const Matrix4x4& mat1, const Matrix4x4& mat2);
extern __device__ Vector::Vector3D NDCtoScreen(const Vector::Vector3D& ndc, int screenWidth, int screenHeight);
extern __device__ Vector::Vector3D cross(const Vector::Vector3D& a, const Vector::Vector3D& b);
extern __device__ Vector::Vector3D ProjectVertex(const Vector::Vector3D& v, const Matrix4x4& viewProj, int screenWidth, int screenHeight);
extern __host__ __device__ Matrix4x4 LookAt(const Vector::Vector3D& eye, const Vector::Vector3D& target, const Vector::Vector3D& up);

//cudaError_t Math_Test(int* c, const int* a, const int* b, unsigned int size);