#pragma once

class Mesh
{
private:
	VertexBuffer* vertexBuffer;
	IndexBuffer* indexBuffer;

public:
	Mesh(void* vertexData, UINT stride, UINT vertexCount,
		void* indexData, UINT indexCount, bool isCPUWrite = false);
	~Mesh();

	void IASet(D3D11_PRIMITIVE_TOPOLOGY primitiveType = D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

	void UpdateVertex(void* data, UINT count);
	void MapVertex(void* data, UINT start, UINT size, UINT offset);

	ID3D11Buffer* GetVertexBuffer() { return vertexBuffer->GetBuffer(); }
};