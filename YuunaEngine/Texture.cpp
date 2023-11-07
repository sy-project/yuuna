#include "header.h"

map<wstring, Texture*> Texture::totalTexture;

Texture::Texture(ID3D11ShaderResourceView* srv, ScratchImage& image)
    : srv(srv), image(move(image))
{
}

Texture::~Texture()
{
    srv->Release();
}

Texture* Texture::Add(wstring file)
{
    if (totalTexture.count(file) > 0)
        return totalTexture[file];

    ScratchImage image;

    wstring extension = Utility::GetExtension(file);

    if (extension == L"tga")
        LoadFromTGAFile(file.c_str(), nullptr, image);
    else if(extension == L"dds")
        LoadFromDDSFile(file.c_str(), DDS_FLAGS_NONE, nullptr, image);
    else
        LoadFromWICFile(file.c_str(), WIC_FLAGS_FORCE_RGB, nullptr, image);

    ID3D11ShaderResourceView* srv;

    V(CreateShaderResourceView(Device::Get()->GetDevice(), image.GetImages(), image.GetImageCount(),
        image.GetMetadata(), &srv));

    totalTexture[file] = new Texture(srv, image);

    return totalTexture[file];
}

Texture* Texture::Load(wstring file)
{
    ScratchImage image;

    wstring extension = Utility::GetExtension(file);

    if (extension == L"tga")
        LoadFromTGAFile(file.c_str(), nullptr, image);
    else if (extension == L"dds")
        LoadFromDDSFile(file.c_str(), DDS_FLAGS_NONE, nullptr, image);
    else
        LoadFromWICFile(file.c_str(), WIC_FLAGS_FORCE_RGB, nullptr, image);

    ID3D11ShaderResourceView* srv;

    V(CreateShaderResourceView(Device::Get()->GetDevice(), image.GetImages(), image.GetImageCount(),
        image.GetMetadata(), &srv));

    if (totalTexture[file] != 0)
        delete totalTexture[file];

    totalTexture[file] = new Texture(srv, image);

    return totalTexture[file];
}

void Texture::Delete()
{
    for (auto texture : totalTexture)
        delete texture.second;
}

void Texture::PSSet(UINT slot)
{
    Device::Get()->GetDeviceContext()->PSSetShaderResources(slot, 1, &srv);
}

void Texture::DSSet(UINT slot)
{
    Device::Get()->GetDeviceContext()->DSSetShaderResources(slot, 1, &srv);
}

vector<Float4> Texture::ReadPixels()
{
    uint8_t* colors = image.GetPixels();
    UINT size = image.GetPixelsSize();

    float scale = 1.0f / 255.0f;

    vector<Float4> result(size / 4);

    for (UINT i = 0; i < result.size() ; i ++)
    {
        result[i].x = colors[i*4 + 0] * scale;
        result[i].y = colors[i*4 + 1] * scale;
        result[i].z = colors[i*4 + 2] * scale;
        result[i].w = colors[i*4 + 3] * scale;
    }

    return result;
}
