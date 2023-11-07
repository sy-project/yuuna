#include "header.h"

PixelShader::PixelShader(wstring file, string entry)
{
	wstring path = L"Shaders/" + file + L".hlsl";

	DWORD flags = D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_DEBUG;

	V(D3DCompileFromFile(path.c_str(), nullptr,
		D3D_COMPILE_STANDARD_FILE_INCLUDE, entry.c_str(), "ps_5_0",
		flags, 0, &blob, nullptr));

	V(Device::Get()->GetDevice()->CreatePixelShader(blob->GetBufferPointer(),
		blob->GetBufferSize(), nullptr, &shader));

	blob->Release();
}

PixelShader::~PixelShader()
{
	shader->Release();
}

void PixelShader::Set()
{
	Device::Get()->GetDeviceContext()->PSSetShader(shader, nullptr, 0);
}
