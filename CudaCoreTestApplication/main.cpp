#include <CoreImporter.h>
#include <CudaImporter.h>
#include <unordered_map>
#include <iostream>
#include <vector>

int main()
{
    std::string winName = "test";

    int WIDTH, HEIGHT;

    WIDTH = 800;
    HEIGHT = 600;

    Core::Init();
    Core::Model::Import3D("../Engine/Resources/miko/sakura_miko.pmx");
    auto a = Core::Model::GetVertex(0);
    auto b = Core::Model::GetObjIdFromName("sakura_miko");
    //Core::Encrypt::EResourceFile("123432.png", "123432.enc");
    SYCUDA::GDevice::DeviceInit(winName, 10, 20, WIDTH, HEIGHT, WS_OVERLAPPEDWINDOW | WS_VISIBLE);

    uint8_t* h_framebuffer = new uint8_t[WIDTH * HEIGHT * 4];

    std::vector<Vertex3D> vvertex;
    Vector::Vector3D p = { 100, 100, 100 };
    Vector::Vector2D uv = { 0,0 };
    Vertex3D v = { p,uv };
    vvertex.push_back(v);
    p = { 100, 110, 200 };
    uv = { 1,0 };
    v = { p,uv };
    vvertex.push_back(v);
    p = { 150, 190, 100 };
    uv = { 0,1 };
    v = { p,uv };
    vvertex.push_back(v);
    p = { 150, 200, 200 };
    uv = { 1,1 };
    v = { p,uv };
    vvertex.push_back(v);

    SYCUDA::GDevice::DeviceInput3DVertex(0, vvertex.at(0), vvertex.at(1), vvertex.at(3));
    SYCUDA::GDevice::DeviceInput3DVertex(0, vvertex.at(2), vvertex.at(0), vvertex.at(3));
    //SYCUDA::GDevice::DeviceInput2DVertex(0, vvertex.at(0), vvertex.at(1), vvertex.at(3));
    //SYCUDA::GDevice::DeviceInput2DVertex(0, vvertex.at(2), vvertex.at(0), vvertex.at(3));
    //SYCUDA::GDevice::DeviceInput2DVertex(1, { 200, 200 }, { 200, 250 }, { 250, 200 });
    SYCUDA::GDevice::DeviceInput2DImageFromEnc(0, Core::Decrypt::DResourceFile("123432.enc"));

    while (true)
    {
        Core::CONTROL::UpdateInput();
        SYCUDA::GDevice::DeviceUpdate2DVertexRotWCenter(0, 0.01f, {500,500});
        if(Core::CONTROL::KeyPress('K'))
            SYCUDA::GDevice::DeviceUpdate2DVertexRot(0, 0.1f);
        SYCUDA::GDevice::DeviceUpdate2DVertexPos(1, {0.01f,0.01f});
        SYCUDA::GDevice::DeviceRender(h_framebuffer);
    }

    delete[] h_framebuffer;
    SYCUDA::GDevice::DeviceDelete();
    Core::End();
    return 0;
}