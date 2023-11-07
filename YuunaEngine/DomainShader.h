#pragma once

class DomainShader : public Shader
{
private:
	friend class Shader;

	ID3D11DomainShader* shader;

	DomainShader(wstring file, string entry);
	~DomainShader();

public:
	virtual void Set() override;
};