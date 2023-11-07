#include "header.h"

DomainShader::DomainShader(wstring file, string entry)
{
	wstring path = L"Shaders/" + file + L".hlsl";

	DWORD flags = D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_DEBUG;

	V(D3DCompileFromFile(path.c_str(), nullptr,
		D3D_COMPILE_STANDARD_FILE_INCLUDE, entry.c_str(), "ds_5_0",
		flags, 0, &blob, nullptr));

	V(Device::Get()->GetDevice()->CreateDomainShader(blob->GetBufferPointer(),
		blob->GetBufferSize(), nullptr, &shader));

	blob->Release();
}

DomainShader::~DomainShader()
{
	shader->Release();
}

void DomainShader::Set()
{
	Device::Get()->GetDeviceContext()->DSSetShader(shader, nullptr, 0);
}
