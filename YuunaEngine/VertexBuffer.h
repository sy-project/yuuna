#pragma once

class VertexBuffer
{
private:
	ID3D11Buffer* buffer;

	UINT stride;
	UINT offset;

public:
	VertexBuffer(void* data, UINT stride, UINT count, bool isCPUWrite = false);	
	~VertexBuffer();

	void IASet(UINT slot = 0);

	void Update(void* data, UINT count);
	void Map(void* data, UINT start, UINT size, UINT offset);

	ID3D11Buffer* GetBuffer() { return buffer; }
};