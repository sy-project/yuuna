#include "header.h"

Mesh::Mesh(void* vertexData, UINT stride, UINT vertexCount, void* indexData, UINT indexCount, bool isCPUWrite)
{
	vertexBuffer = new VertexBuffer(vertexData, stride, vertexCount, isCPUWrite);
	indexBuffer = new IndexBuffer(indexData, indexCount);
}

Mesh::~Mesh()
{
	delete vertexBuffer;
	delete indexBuffer;
}

void Mesh::IASet(D3D11_PRIMITIVE_TOPOLOGY primitiveType)
{
	vertexBuffer->IASet();
	indexBuffer->IASet();
	Device::Get()->GetDeviceContext()->IASetPrimitiveTopology(primitiveType);
}

void Mesh::UpdateVertex(void* data, UINT count)
{
	vertexBuffer->Update(data, count);
}

void Mesh::MapVertex(void* data, UINT start, UINT size, UINT offset)
{
	vertexBuffer->Map(data, start, size, offset);
}
