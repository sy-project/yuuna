#pragma once

class Material
{
public:
	string name;

private:
	class MaterialBuffer : public ConstBuffer
	{
	public:
		struct Data
		{
			Float4 diffuse;
			Float4 specular;
			Float4 ambient;
			Float4 emissive;

			float shininess;

			int hasDiffuseMap;
			int hasSpecularMap;
			int hasNormalMap;
		}data;

		MaterialBuffer() : ConstBuffer(&data, sizeof(Data))
		{
			data.diffuse = { 1, 1, 1, 1 };
			data.specular = { 1, 1, 1, 1 };
			data.ambient = { 1, 1, 1, 1 };
			data.emissive = { 0.3f, 0.3f, 0.3f, 0.1f };
			data.shininess = 30;

			data.hasDiffuseMap = 0;
			data.hasSpecularMap = 0;
			data.hasNormalMap = 0;
		}
	};	

	VertexShader* vertexShader;
	PixelShader* pixelShader;

	MaterialBuffer* buffer;

	Texture* diffuseMap;
	Texture* specularMap;
	Texture* normalMap;
	
public:
	Material();
	Material(wstring file);
	Material(VertexShader* vertexShader, PixelShader* pixelShader);
	~Material();

	void Set();

	void SetShader(wstring file);

	void SetDiffuseMap(wstring file);
	void SetSpecularMap(wstring file);
	void SetNormalMap(wstring file);

	MaterialBuffer* GetBuffer() { return buffer; }
};