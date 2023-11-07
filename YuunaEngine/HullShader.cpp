#include "header.h"

HullShader::HullShader(wstring file, string entry)
{
	wstring path = L"Shaders/" + file + L".hlsl";

	DWORD flags = D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_DEBUG;

	V(D3DCompileFromFile(path.c_str(), nullptr,
		D3D_COMPILE_STANDARD_FILE_INCLUDE, entry.c_str(), "hs_5_0",
		flags, 0, &blob, nullptr));

	V(Device::Get()->GetDevice()->CreateHullShader(blob->GetBufferPointer(),
		blob->GetBufferSize(), nullptr, &shader));

	blob->Release();
}

HullShader::~HullShader()
{
	shader->Release();
}

void HullShader::Set()
{
	Device::Get()->GetDeviceContext()->HSSetShader(shader, nullptr, 0);
}
