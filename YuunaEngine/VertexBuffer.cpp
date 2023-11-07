#include "header.h"

VertexBuffer::VertexBuffer(void* data, UINT stride, UINT count, bool isCPUWrite)
	: stride(stride), offset(0)
{
	D3D11_BUFFER_DESC desc = {};
	if (isCPUWrite)
	{
		desc.Usage = D3D11_USAGE_DYNAMIC;
		desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
	}else
		desc.Usage = D3D11_USAGE_DEFAULT;

	desc.ByteWidth = stride * count;
	desc.BindFlags = D3D11_BIND_VERTEX_BUFFER;

	D3D11_SUBRESOURCE_DATA initData = {};
	initData.pSysMem = data;

	V(Device::Get()->GetDevice()->CreateBuffer(&desc, &initData, &buffer));
}

VertexBuffer::~VertexBuffer()
{
	buffer->Release();
}

void VertexBuffer::IASet(UINT slot)
{
	Device::Get()->GetDeviceContext()->IASetVertexBuffers(slot, 1, &buffer, &stride, &offset);
}

void VertexBuffer::Update(void* data, UINT count)
{
	Device::Get()->GetDeviceContext()->UpdateSubresource(buffer, 0, nullptr, data, stride, count);
}

void VertexBuffer::Map(void* data, UINT start, UINT size, UINT offset)
{
	D3D11_MAPPED_SUBRESOURCE subResource;
	subResource.pData = data;

	Device::Get()->GetDeviceContext()->Map(buffer, 0, D3D11_MAP_WRITE_NO_OVERWRITE, 0, &subResource);
	memcpy((BYTE*)subResource.pData + offset, (BYTE*)data + start, size);
	Device::Get()->GetDeviceContext()->Unmap(buffer, 0);
}

