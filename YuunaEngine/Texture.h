#pragma once

HRESULT CreateWICTextureFromMemory(_In_ ID3D11Device* d3dDevice,
	_In_opt_ ID3D11DeviceContext* d3dContext,
	_In_bytecount_(wicDataSize) const uint8_t* wicData,
	_In_ size_t wicDataSize,
	_Out_opt_ ID3D11Resource** texture,
	_Out_opt_ ID3D11ShaderResourceView** textureView,
	_In_ size_t maxsize = 0
);

HRESULT CreateWICTextureFromFile(_In_ ID3D11Device* d3dDevice,
	_In_opt_ ID3D11DeviceContext* d3dContext,
	_In_z_ const wchar_t* szFileName,
	_Out_opt_ ID3D11Resource** texture,
	_Out_opt_ ID3D11ShaderResourceView** textureView,
	_In_ size_t maxsize = 0
);

class Texture
{
private:
	ScratchImage image;

	ID3D11ShaderResourceView* srv;

	static map<wstring, Texture*> totalTexture;

	Texture(ID3D11ShaderResourceView* srv, ScratchImage& image);
	~Texture();

public:
	static Texture* Add(wstring file);
	static Texture* Load(wstring file);
	static void Delete();

	void PSSet(UINT slot);
	void DSSet(UINT slot);

	vector<Float4> ReadPixels();

	UINT Width() { return image.GetMetadata().width; }
	UINT Height() { return image.GetMetadata().height; }

	ID3D11ShaderResourceView* GetSRV() { return srv; }
};