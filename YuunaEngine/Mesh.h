#pragma once

template<typename T>
inline void SafeRelease(T*& x) {
	if (x) {
		x->Release();
		x = nullptr;
	}
}

struct VERTEX {
	FLOAT X, Y, Z;
	XMFLOAT2 texcoord;
};

struct _Texture {
	std::string type;
	std::string path;
	ID3D11ShaderResourceView* texture;

	void sfRelease() {
		SafeRelease(texture);
	}
};

class Mesh
{
public:
	std::vector<VERTEX> vertices_;
	std::vector<UINT> indices_;
	std::vector<_Texture> textures_;

	Mesh(const std::vector<VERTEX>& vertices, const std::vector<UINT>& indices, const std::vector<_Texture>& textures);
	Mesh(void* vertexData, UINT stride, UINT vertexCount,
		void* indexData, UINT indexCount, bool isCPUWrite = false);
	void deleteMesh();
	void Draw();
	void Close();

	void IASet(D3D11_PRIMITIVE_TOPOLOGY primitiveType = D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

	void UpdateVertex(void* data, UINT count);
	void MapVertex(void* data, UINT start, UINT size, UINT offset);

	ID3D11Buffer* GetVertexBuffer() { return vertexBuffer->GetBuffer(); }

private:
	//Collider Data
	VertexBuffer* vertexBuffer;
	IndexBuffer* indexBuffer;
	// Render data
	ID3D11Buffer* VertexBuffer_, * IndexBuffer_;
	void setupMesh();
};