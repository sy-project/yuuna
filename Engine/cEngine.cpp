#include "framework.h"
#include "cEngine.h"
#include "SafeRelease.hpp"

cEngine::cEngine()
{
	cModelManager::Get();
	m_ck = false;
	D3D11_VIEWPORT viewport;
	ZeroMemory(&viewport, sizeof(D3D11_VIEWPORT));

	viewport.TopLeftX = 0;
	viewport.TopLeftY = 0;
	viewport.MinDepth = 0.0f;
	viewport.MaxDepth = 1.0f;
	viewport.Width = WIN_X;
	viewport.Height = WIN_Y;

	cDevice::Get()->GetDeviceContext()->RSSetViewports(1, &viewport);

	ID3DBlob* VS, * PS;
	m_pxShader = L"PixelShader.hlsl";
	m_vxShader = L"VertexShader.hlsl";

	if (FAILED(CompileShaderFromFile((WCHAR*)m_vxShader.c_str(), 0, "main", "vs_4_0", &VS)))
		Throwanerror("Failed to compile shader from file");
	if (FAILED(CompileShaderFromFile((WCHAR*)m_pxShader.c_str(), 0, "main", "ps_4_0", &PS)))
		Throwanerror("Failed to compile shader from file ");
	
	cDevice::Get()->GetDevice()->CreateVertexShader(VS->GetBufferPointer(), VS->GetBufferSize(), nullptr, &pVS);
	cDevice::Get()->GetDevice()->CreatePixelShader(PS->GetBufferPointer(), PS->GetBufferSize(), nullptr, &pPS);

	D3D11_INPUT_ELEMENT_DESC ied[] =
	{
		{ "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
		{ "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0 }
	};

	cDevice::Get()->GetDevice()->CreateInputLayout(ied, 2, VS->GetBufferPointer(), VS->GetBufferSize(), &pLayout);
	cDevice::Get()->GetDeviceContext()->IASetInputLayout(pLayout);

	m_Projection = XMMatrixPerspectiveFovLH(XM_PIDIV4, WIN_X / (float)WIN_Y, 0.01f, 1000.0f);

	D3D11_BUFFER_DESC bd;
	ZeroMemory(&bd, sizeof(bd));

	bd.Usage = D3D11_USAGE_DEFAULT;
	bd.ByteWidth = sizeof(ConstantBuffer);
	bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
	bd.CPUAccessFlags = 0;

	auto hr = cDevice::Get()->GetDevice()->CreateBuffer(&bd, nullptr, &pConstantBuffer);
	if (FAILED(hr))
		Throwanerror("Constant buffer couldn't be created");

	D3D11_SAMPLER_DESC sampDesc;
	ZeroMemory(&sampDesc, sizeof(sampDesc));
	sampDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
	sampDesc.AddressU = D3D11_TEXTURE_ADDRESS_WRAP;
	sampDesc.AddressV = D3D11_TEXTURE_ADDRESS_WRAP;
	sampDesc.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;
	sampDesc.ComparisonFunc = D3D11_COMPARISON_NEVER;
	sampDesc.MinLOD = 0;
	sampDesc.MaxLOD = D3D11_FLOAT32_MAX;

	hr = cDevice::Get()->GetDevice()->CreateSamplerState(&sampDesc, &TexSamplerState);
	if (FAILED(hr))
		Throwanerror("Texture sampler state couldn't be created");

	Vector4 Eye = XMVectorSet(0.0f, 5.0f, -30.0f, 0.0f);
	Vector4 At = XMVectorSet(0.0f, 10.0f, 0.0f, 0.0f);
	Vector4 Up = XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f);
	m_View = XMMatrixLookAtLH(Eye, At, Up);

	ModelLoader* m = new ModelLoader;
	for (int i = 0; i < 1; i++)
	{
		switch (i)
		{
		case 0:
			m_ModelPath = "Resources\\miko\\sakura_miko.pmx";
			break;
		case 1:
			m_ModelPath = "Resources\\sora\\sora.pmx";
			break;
		default:
			break;
		}
		if (!m->Load(cDevice::Get()->GetDevice(), cDevice::Get()->GetDeviceContext(), m_ModelPath))
			Throwanerror("Model couldn't be loaded");
		m_models.insert({ i, m });
	}
}

cEngine::~cEngine()
{
	SafeRelease(pVS);
	SafeRelease(pPS);
	SafeRelease(pLayout);
	SafeRelease(pConstantBuffer);
	SafeRelease(TexSamplerState);
}


void cEngine::Update()
{
	const int arraySize = 17;
	const int a[arraySize] = {	1, 2, 3, 4, 5,
								6, 7, 8, 9, 10,
								11, 12, 13, 14, 15,
								16, 17};
	const int b[arraySize] = { 10, 20, 30, 40, 50,
								60, 70, 80, 90 , 100,
								110, 120, 130, 140, 150,
								160, 170};
	int c[arraySize] = { 0 };

	cuda_dll::math::CudaMath_test(c, a, b, arraySize);
	cuda_text = "[";
	cuda_text += to_string(a[0]) + "/";
	cuda_text += to_string(a[1]) + "/";
	cuda_text += to_string(a[2]) + "/";
	cuda_text += to_string(a[3]) + "/";
	cuda_text += to_string(a[4]) + "/";
	cuda_text += to_string(a[5]) + "/";
	cuda_text += to_string(a[6]) + "/";
	cuda_text += to_string(a[7]) + "/";
	cuda_text += to_string(a[8]) + "/";
	cuda_text += to_string(a[9]) + "/";
	cuda_text += to_string(a[10]) + "/";
	cuda_text += to_string(a[11]) + "/";
	cuda_text += to_string(a[12]) + "/";
	cuda_text += to_string(a[13]) + "/";
	cuda_text += to_string(a[14]) + "/";
	cuda_text += to_string(a[15]) + "/";
	cuda_text += to_string(a[16]) + "] + \n";

	cuda_text += "[";
	cuda_text += to_string(b[0]) + "/";
	cuda_text += to_string(b[1]) + "/";
	cuda_text += to_string(b[2]) + "/";
	cuda_text += to_string(b[3]) + "/";
	cuda_text += to_string(b[4]) + "/";
	cuda_text += to_string(b[5]) + "/";
	cuda_text += to_string(b[6]) + "/";
	cuda_text += to_string(b[7]) + "/";
	cuda_text += to_string(b[8]) + "/";
	cuda_text += to_string(b[9]) + "/";
	cuda_text += to_string(b[10]) + "/";
	cuda_text += to_string(b[11]) + "/";
	cuda_text += to_string(b[12]) + "/";
	cuda_text += to_string(b[13]) + "/";
	cuda_text += to_string(b[14]) + "/";
	cuda_text += to_string(b[15]) + "/";
	cuda_text += to_string(b[16]) + "] = \n";

	cuda_text += "[";
	cuda_text += to_string(c[0]) + "/";
	cuda_text += to_string(c[1]) + "/";
	cuda_text += to_string(c[2]) + "/";
	cuda_text += to_string(c[3]) + "/";
	cuda_text += to_string(c[4]) + "/";
	cuda_text += to_string(c[5]) + "/";
	cuda_text += to_string(c[6]) + "/";
	cuda_text += to_string(c[7]) + "/";
	cuda_text += to_string(c[8]) + "/";
	cuda_text += to_string(c[9]) + "/";
	cuda_text += to_string(c[10]) + "/";
	cuda_text += to_string(c[11]) + "/";
	cuda_text += to_string(c[12]) + "/";
	cuda_text += to_string(c[13]) + "/";
	cuda_text += to_string(c[14]) + "/";
	cuda_text += to_string(c[15]) + "/";
	cuda_text += to_string(c[16]) + "]";
	static float t = 0.0f;
	static ULONGLONG timeStart = 0;
	ULONGLONG timeCur = GetTickCount64();
	if (timeStart == 0)
		timeStart = timeCur;
	t = (timeCur - timeStart) / 1000.0f;

	cDevice::Get()->GetDeviceContext()->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

	m_World = XMMatrixRotationY(-t);

	ConstantBuffer cb;
	cb.mWorld = XMMatrixTranspose(m_World);
	cb.mView = XMMatrixTranspose(m_View);
	cb.mProjection = XMMatrixTranspose(m_Projection);
	cDevice::Get()->GetDeviceContext()->UpdateSubresource(pConstantBuffer, 0, nullptr, &cb, 0, 0);

	cDevice::Get()->GetDeviceContext()->VSSetShader(pVS, 0, 0);
	cDevice::Get()->GetDeviceContext()->VSSetConstantBuffers(0, 1, &pConstantBuffer);
	cDevice::Get()->GetDeviceContext()->PSSetShader(pPS, 0, 0);
	cDevice::Get()->GetDeviceContext()->PSSetSamplers(0, 1, &TexSamplerState);
}

void cEngine::PreRender()
{
}

void cEngine::Render()
{
	for (int i = 0; i < m_models.size(); i++)
	{
		m_models.at(i)->Draw(cDevice::Get()->GetDeviceContext());
	}
}

void cEngine::PostRender()
{
	if(cImGuiManager::Get()->OpenImGuiWindow("test"))
	{
		ImGui::Text(cuda_text.c_str());
	}
	ImGui::End();
}

void cEngine::Throwanerror(LPCSTR errormessage)
{
	throw std::runtime_error(errormessage);
}

HRESULT cEngine::CompileShaderFromFile(LPCWSTR pFileName, const D3D_SHADER_MACRO* pDefines, LPCSTR pEntryPoint, LPCSTR pShaderModel, ID3DBlob** ppBytecodeBlob)
{
	UINT compileFlags = D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_PACK_MATRIX_COLUMN_MAJOR;

#ifdef _DEBUG
	compileFlags |= D3DCOMPILE_DEBUG;
#endif

	ID3DBlob* pErrorBlob = nullptr;

	HRESULT result = D3DCompileFromFile(pFileName, pDefines, D3D_COMPILE_STANDARD_FILE_INCLUDE, pEntryPoint, pShaderModel, compileFlags, 0, ppBytecodeBlob, &pErrorBlob);
	if (FAILED(result))
	{
		if (pErrorBlob != nullptr)
			OutputDebugStringA((LPCSTR)pErrorBlob->GetBufferPointer());
	}

	if (pErrorBlob != nullptr)
		pErrorBlob->Release();

	return result;
}

