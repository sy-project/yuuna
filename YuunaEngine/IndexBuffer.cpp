#include "header.h"

IndexBuffer::IndexBuffer(void* data, UINT count)
{
	D3D11_BUFFER_DESC desc = {};
	desc.Usage = D3D11_USAGE_DEFAULT;
	desc.ByteWidth = sizeof(UINT) * count;
	desc.BindFlags = D3D11_BIND_INDEX_BUFFER;

	D3D11_SUBRESOURCE_DATA initData = {};
	initData.pSysMem = data;

	V(Device::Get()->GetDevice()->CreateBuffer(&desc, &initData, &buffer));
}

IndexBuffer::~IndexBuffer()
{
	buffer->Release();
}

void IndexBuffer::IASet()
{
	Device::Get()->GetDeviceContext()->IASetIndexBuffer(buffer, DXGI_FORMAT_R32_UINT, 0);
}
