#pragma once

class VertexShader : public Shader
{
private:
	friend class Shader;

	ID3D11VertexShader* shader;
	ID3D11InputLayout* inputLayout;

	ID3D11ShaderReflection* reflection;
	
	VertexShader(wstring file, string entry);
	~VertexShader();

public:	
	virtual void Set() override;

private:
	void CreateInputLayout();
};