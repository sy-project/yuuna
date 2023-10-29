#pragma once

#ifdef COMMON_EXPORTS
#define DEVICE_DECLSPEC __declspec(dllexport)
#else
#define DEVICE_DECLSPEC __declspec(dllimport)
#endif

class Device : public Singleton<Device> {
private:
	friend class Singleton;
	Device();
	~Device();

private:
	ID3D11Device* device;
	ID3D11DeviceContext* deviceContext;

	IDXGISwapChain* swapChain;
	ID3D11RenderTargetView* renderTargetView;
	ID3D11DepthStencilView* depthStencilView;

public:
	void CreateDeviceAndSwapChain(int WIN_X, int WIN_Y, HWND hWnd);
	void CreateBackBuffer(int WIN_X, int WIN_Y);
	void SetRenderTarget();
	void Clear(Float4 color = Float4(0.1f, 0.1f, 0.125f, 1.0f));
	void Present();

	ID3D11Device* GetDevice() { return device; }
	ID3D11DeviceContext* GetDeviceContext() { return deviceContext; }
	IDXGISwapChain* GetSwapChain() { return swapChain; }
};