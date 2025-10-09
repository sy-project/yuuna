#define NOMINMAX
#include <CoreImporter.h>
#include <CudaImporter.h>
#include <unordered_map>
#include <iostream>
#include <vector>

#define WIDTH 1920 //1920
#define HEIGHT 1080 //1080

int main()
{
    Core::Init();
    sWindow s;
    s.winName = "test";
    s.pos = { 0,0 };
    s.size = { WIDTH,HEIGHT };
    s.dwStyle = WS_OVERLAPPEDWINDOW | WS_VISIBLE;

    SYCUDA::Check_Cuda(s);
    SYCUDA::GDevice::DeviceInit(s);

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

    SYCUDA::GDevice::DeviceInput2DVertex(s, 0, vvertex.at(0), vvertex.at(1), vvertex.at(3));
    SYCUDA::GDevice::DeviceInput2DVertex(s, 0, vvertex.at(2), vvertex.at(0), vvertex.at(3));
    //SYCUDA::GDevice::DeviceInput2DVertex(1, { 200, 200 }, { 200, 250 }, { 250, 200 });
    SYCUDA::GDevice::DeviceInput2DImageFromEnc(s, 0, 0, Core::Decrypt::DResourceFile("123432.enc"));

    //Core::Model::Import3D("../Engine/Resources/miko/sakura_miko.pmx");
    //auto objid = Core::Model::GetObjIdFromName("sakura_miko");
    //auto v = Core::Model::GetTriangle(Core::Model::GetObjIdFromName("sakura_miko"));
    //auto Tex = Core::Model::GetTexture(Core::Model::GetObjIdFromName("sakura_miko"));
    //SYCUDA::GDevice::DeviceInput3DTriangle(objid, v);
    //for (int i = 0; i < Tex.size(); i++)
    //{
    //    SYCUDA::GDevice::DeviceInput3DTex(objid, Tex.at(i).texId, Tex.at(i).img, Tex.at(i).size.x, Tex.at(i).size.y);
    //}

    //SYCUDA::GDevice::DeviceInput3DTex(objid, Tex. Tex.img, Tex.size.x, Tex.size.y);

    int frame = 0;
    auto start = std::chrono::high_resolution_clock::now();
    while (true)
    {
        frame++;
        auto now = std::chrono::high_resolution_clock::now();

        Core::CONTROL::UpdateInput();
        SYCUDA::GDevice::DeviceUpdate2DVertexRotWCenter(s, 0, 0.01f, { 500,500 });
        //if (Core::CONTROL::KeyPress('K'))
        //    SYCUDA::GDevice::DeviceUpdate2DVertexRot(0, 0.1f);
        //SYCUDA::GDevice::DeviceUpdate2DVertexPos(1, { 0.01f,0.01f });
        SYCUDA::GDevice::DeviceRender(s);

        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start);
        if (elapsed.count() >= 1) {
            std::string logTemp = "FPS: " + std::to_string(frame);
            std::cout << logTemp << "\n";
            Core::Log::WriteLog(logTemp);
            frame = 0;
            start = now;
        }
    }

    SYCUDA::GDevice::DeviceDelete();
    Core::End();
    return 0;
}