#pragma once

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