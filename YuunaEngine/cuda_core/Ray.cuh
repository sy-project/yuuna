#include <cmath>
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>
#include "cuda_header.h"
#include <Types.h>

class Ray
{
public:
	Ray(Vector::Vector3D _origin);
	~Ray();

	__device__ bool hit(Vector::Vector3D _ori, Vector::Vector3D _dir);

private:
	Vector::Vector3D origin;
	Vector::Vector3D direction;
};
