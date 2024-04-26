#pragma once
class cDevice : public Singleton<cDevice>
{
private:
	friend class Singleton;

	cDevice();
	~cDevice();

private:
	ID3D11Device* device;
	ID3D11DeviceContext* deviceContext;

	IDXGISwapChain* swapChain;
	ID3D11RenderTargetView* renderTargetView;
	ID3D11DepthStencilView* depthStencilView;

public:
	void CreateDeviceAndSwapChain();
	void CreateBackBuffer();
	void SetRenderTarget();
	void Clear(Float4 color = Float4(0.1f, 0.1f, 0.125f, 1.0f));
	void Present();

	ID3D11Device* GetDevice() { return device; }
	ID3D11DeviceContext* GetDeviceContext() { return deviceContext; }
	IDXGISwapChain* GetSwapChain() { return swapChain; }
};

