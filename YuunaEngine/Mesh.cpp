#include "header.h"

Mesh::Mesh(const std::vector<VERTEX>& vertices, const std::vector<UINT>& indices, const std::vector<_Texture>& textures) :
	vertices_(vertices),
	indices_(indices),
	textures_(textures),
	VertexBuffer_(nullptr),
	IndexBuffer_(nullptr)
{
	this->setupMesh();
	//vertexBuffer = new VertexBuffer(vertexData, stride, vertexCount, isCPUWrite);
	//indexBuffer = new IndexBuffer(indexData, indexCount);
}

Mesh::Mesh(void* vertexData, UINT stride, UINT vertexCount, void* indexData, UINT indexCount, bool isCPUWrite)
{
    vertexBuffer = new VertexBuffer(vertexData, stride, vertexCount, isCPUWrite);
    indexBuffer = new IndexBuffer(indexData, indexCount);
}

void Mesh::deleteMesh()
{
    delete vertexBuffer;
    delete indexBuffer;
}

void Mesh::Draw()
{
    UINT stride = sizeof(VERTEX);
    UINT offset = 0;

    Device::Get()->GetDeviceContext()->IASetVertexBuffers(0, 1, &VertexBuffer_, &stride, &offset);
    Device::Get()->GetDeviceContext()->IASetIndexBuffer(IndexBuffer_, DXGI_FORMAT_R32_UINT, 0);

    if(textures_.size()>0)
    Device::Get()->GetDeviceContext()->PSSetShaderResources(0, 1, &textures_[0].texture);

    Device::Get()->GetDeviceContext()->DrawIndexed(static_cast<UINT>(indices_.size()), 0, 0);
}

void Mesh::Close()
{
    SafeRelease(VertexBuffer_);
    SafeRelease(IndexBuffer_);
}

void Mesh::setupMesh()
{
    HRESULT hr;

    D3D11_BUFFER_DESC vbd;
    vbd.Usage = D3D11_USAGE_IMMUTABLE;
    vbd.ByteWidth = static_cast<UINT>(sizeof(VERTEX) * vertices_.size());
    vbd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
    vbd.CPUAccessFlags = 0;
    vbd.MiscFlags = 0;

    D3D11_SUBRESOURCE_DATA initData;
    initData.pSysMem = &vertices_[0];

    hr = Device::Get()->GetDevice()->CreateBuffer(&vbd, &initData, &VertexBuffer_);
    if (FAILED(hr)) {
        Close();
        throw std::runtime_error("Failed to create vertex buffer.");
    }

    D3D11_BUFFER_DESC ibd;
    ibd.Usage = D3D11_USAGE_IMMUTABLE;
    ibd.ByteWidth = static_cast<UINT>(sizeof(UINT) * indices_.size());
    ibd.BindFlags = D3D11_BIND_INDEX_BUFFER;
    ibd.CPUAccessFlags = 0;
    ibd.MiscFlags = 0;

    initData.pSysMem = &indices_[0];

    hr = Device::Get()->GetDevice()->CreateBuffer(&ibd, &initData, &IndexBuffer_);
    if (FAILED(hr)) {
        Close();
        throw std::runtime_error("Failed to create index buffer.");
    }
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
