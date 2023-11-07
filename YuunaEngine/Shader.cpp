#include "header.h"

map<wstring, Shader*> Shader::totalShader;

VertexShader* Shader::AddVS(wstring file, string entry)
{
    wstring key = file + Utility::ToWString(entry);

    if (totalShader.count(key) > 0)
        return (VertexShader*)totalShader[key];

    totalShader[key] = new VertexShader(file, entry);

    return (VertexShader*)totalShader[key];
}

PixelShader* Shader::AddPS(wstring file, string entry)
{
    wstring key = file + Utility::ToWString(entry);

    if (totalShader.count(key) > 0)
        return (PixelShader*)totalShader[key];

    totalShader[key] = new PixelShader(file, entry);

    return (PixelShader*)totalShader[key];
}

ComputeShader* Shader::AddCS(wstring file, string entry)
{
    wstring key = file + Utility::ToWString(entry);

    if (totalShader.count(key) > 0)
        return (ComputeShader*)totalShader[key];

    totalShader[key] = new ComputeShader(file, entry);

    return (ComputeShader*)totalShader[key];
}

HullShader* Shader::AddHS(wstring file, string entry)
{
    wstring key = file + Utility::ToWString(entry);

    if (totalShader.count(key) > 0)
        return (HullShader*)totalShader[key];

    totalShader[key] = new HullShader(file, entry);

    return (HullShader*)totalShader[key];
}

DomainShader* Shader::AddDS(wstring file, string entry)
{
    wstring key = file + Utility::ToWString(entry);

    if (totalShader.count(key) > 0)
        return (DomainShader*)totalShader[key];

    totalShader[key] = new DomainShader(file, entry);

    return (DomainShader*)totalShader[key];
}

GeometryShader* Shader::AddGS(wstring file, string entry)
{
    wstring key = file + Utility::ToWString(entry);

    if (totalShader.count(key) > 0)
        return (GeometryShader*)totalShader[key];

    totalShader[key] = new GeometryShader(file, entry);

    return (GeometryShader*)totalShader[key];
}

void Shader::Delete()
{
    for (auto shader : totalShader)
        delete shader.second;
}
