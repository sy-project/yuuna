#include "header.h"

GeometryShader::GeometryShader(wstring file, string entry)
{
	wstring path = L"Shaders/" + file + L".hlsl";

	DWORD flags = D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_DEBUG;

	V(D3DCompileFromFile(path.c_str(), nullptr,
		D3D_COMPILE_STANDARD_FILE_INCLUDE, entry.c_str(), "gs_5_0",
		flags, 0, &blob, nullptr));

	V(Device::Get()->GetDevice()->CreateGeometryShader(blob->GetBufferPointer(),
		blob->GetBufferSize(), nullptr, &shader));

	blob->Release();
}

GeometryShader::~GeometryShader()
{
	shader->Release();
}

void GeometryShader::Set()
{
	Device::Get()->GetDeviceContext()->GSSetShader(shader, nullptr, 0);
}
