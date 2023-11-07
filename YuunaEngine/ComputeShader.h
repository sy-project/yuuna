#pragma once

class ComputeShader : public Shader
{
private:
	friend class Shader;

	ID3D11ComputeShader* shader;

	ComputeShader(wstring file, string entry);
	~ComputeShader();

public:
	virtual void Set() override;
};