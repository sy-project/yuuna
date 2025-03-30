#include <CoreImporter.h>
#include <CudaImporter.h>
#include <unordered_map>
#include <iostream>
#include <vector>

void ThreadFunc(uint8_t* h_framebuffer)
{
    while (true)
    {
        this_thread::sleep_for(std::chrono::milliseconds(100));
        SYCUDA::GDevice::DeviceRender(h_framebuffer);
    }
}
int main()
{
    std::string winName = "test";

    int WIDTH, HEIGHT;

    WIDTH = 800;
    HEIGHT = 600;

    Core::Init();
    SYCUDA::GDevice::DeviceInit(winName, 10, 20, WIDTH, HEIGHT, WS_OVERLAPPEDWINDOW | WS_VISIBLE);

    uint8_t* h_framebuffer = new uint8_t[WIDTH * HEIGHT * 4];

#ifdef RENDER2D
    std::vector<Vertex2D> vvertex;
    Vector::Vector2D p = { 100, 100 };
    Vector::Vector2D uv = { 0,0 };
    Vertex2D v = { p,uv };
    vvertex.push_back(v);
    p = { 100, 200 };
    uv = { 1,0 };
    v = { p,uv };
    vvertex.push_back(v);
    p = { 200, 100 };
    uv = { 0,1 };
    v = { p,uv };
    vvertex.push_back(v);
    p = { 200, 200 };
    uv = { 1,1 };
    v = { p,uv };
    vvertex.push_back(v);

    SYCUDA::GDevice::DeviceInput2DVertex(0, vvertex.at(0), vvertex.at(1), vvertex.at(3));
    SYCUDA::GDevice::DeviceInput2DVertex(0, vvertex.at(2), vvertex.at(0), vvertex.at(3));
    //SYCUDA::GDevice::DeviceInput2DVertex(1, { 200, 200 }, { 200, 250 }, { 250, 200 });
    SYCUDA::GDevice::DeviceInput2DImageFromEnc(0, Core::Decrypt::DResourceFile("123432.enc"));

    while (true)
    {
        Core::CONTROL::UpdateInput();
        SYCUDA::GDevice::DeviceUpdate2DVertexRotWCenter(0, 0.01f, { 500,500 });
        if (Core::CONTROL::KeyPress('K'))
            SYCUDA::GDevice::DeviceUpdate2DVertexRot(0, 0.1f);
        SYCUDA::GDevice::DeviceUpdate2DVertexPos(1, { 0.01f,0.01f });
        SYCUDA::GDevice::DeviceRender(h_framebuffer);
    }
#else
    Core::Model::Import3D("../Engine/Resources/miko/sakura_miko.pmx");
    auto objid = Core::Model::GetObjIdFromName("sakura_miko");
    auto v = Core::Model::GetTriangle(Core::Model::GetObjIdFromName("sakura_miko"));
    auto Tex = Core::Model::GetTexture(Core::Model::GetObjIdFromName("sakura_miko"));
    SYCUDA::GDevice::DeviceInput3DTriangle(objid, v);
    for (int i = 0; i < Tex.size(); i++)
    {
        SYCUDA::GDevice::DeviceInput3DTex(objid, Tex.at(i).texId, Tex.at(i).img, Tex.at(i).size.x, Tex.at(i).size.y);
    }
    //SYCUDA::GDevice::DeviceInput3DTex(objid, Tex. Tex.img, Tex.size.x, Tex.size.y);
    thread a(ThreadFunc, h_framebuffer);
    a.detach();
    while (true)
    {
        this_thread::sleep_for(std::chrono::milliseconds(100));
    }
#endif
    delete[] h_framebuffer;
    SYCUDA::GDevice::DeviceDelete();
    Core::End();
    return 0;
}