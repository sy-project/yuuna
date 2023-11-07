#include "header.h"

ConstBuffer::ConstBuffer(void* data, UINT dataSize)
	: data(data), dataSize(dataSize)
{
	D3D11_BUFFER_DESC desc = {};
	desc.Usage = D3D11_USAGE_DYNAMIC;
	desc.ByteWidth = dataSize;
	desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
	desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;	

	V(Device::Get()->GetDevice()->CreateBuffer(&desc, nullptr, &buffer));
}

ConstBuffer::~ConstBuffer()
{
	buffer->Release();
}

void ConstBuffer::Update()
{
	Device::Get()->GetDeviceContext()->Map(buffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &subResource);
	memcpy(subResource.pData, data, dataSize);
	Device::Get()->GetDeviceContext()->Unmap(buffer, 0);
}

void ConstBuffer::SetVSBuffer(UINT slot)
{
	Update();
	Device::Get()->GetDeviceContext()->VSSetConstantBuffers(slot, 1, &buffer);
}

void ConstBuffer::SetPSBuffer(UINT slot)
{
	Update();
	Device::Get()->GetDeviceContext()->PSSetConstantBuffers(slot, 1, &buffer);
}

void ConstBuffer::SetCSBuffer(UINT slot)
{
	Update();
	Device::Get()->GetDeviceContext()->CSSetConstantBuffers(slot, 1, &buffer);
}

void ConstBuffer::SetHSBuffer(UINT slot)
{
	Update();
	Device::Get()->GetDeviceContext()->HSSetConstantBuffers(slot, 1, &buffer);
}

void ConstBuffer::SetDSBuffer(UINT slot)
{
	Update();
	Device::Get()->GetDeviceContext()->DSSetConstantBuffers(slot, 1, &buffer);
}

void ConstBuffer::SetGSBuffer(UINT slot)
{
	Update();
	Device::Get()->GetDeviceContext()->GSSetConstantBuffers(slot, 1, &buffer);
}
