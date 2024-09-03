#pragma once
#include "ModelLoader.h"
struct ConstantBuffer {
	Matrix mWorld;
	Matrix mView;
	Matrix mProjection;
};

class cEngine
{
private:
	wstring m_pxShader;
	wstring m_vxShader;

	string m_ModelPath;

	map<int, ModelLoader*> m_models;

	ID3D11VertexShader* pVS;
	ID3D11PixelShader* pPS;
	ID3D11InputLayout* pLayout;

	Matrix m_World;
	Matrix m_View;
	Matrix m_Projection;

	ID3D11Buffer* pConstantBuffer;
	ID3D11SamplerState* TexSamplerState;

	bool m_ck;

	string cuda_text;

public:
	cEngine();
	~cEngine();

	void Update();
	void PreRender();
	void Render();
	void PostRender();

private:
	void Throwanerror(LPCSTR errormessage);
	HRESULT	CompileShaderFromFile(LPCWSTR pFileName, const D3D_SHADER_MACRO* pDefines, LPCSTR pEntryPoint, LPCSTR pShaderModel, ID3DBlob** ppBytecodeBlob);
};

