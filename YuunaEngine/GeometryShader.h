#pragma once

class GeometryShader : public Shader
{
private:
	friend class Shader;

	ID3D11GeometryShader* shader;

	GeometryShader(wstring file, string entry);
	~GeometryShader();

public:
	virtual void Set() override;
};