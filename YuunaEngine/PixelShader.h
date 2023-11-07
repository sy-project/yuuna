#pragma once

class PixelShader : public Shader
{
private:
	friend class Shader;

	ID3D11PixelShader* shader;	

	PixelShader(wstring file, string entry);
	~PixelShader();

public:
	virtual void Set() override;
};