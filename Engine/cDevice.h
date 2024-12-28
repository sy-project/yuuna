#pragma once
class cDevice : public Singleton<cDevice>
{
private:
	friend class Singleton;

	cDevice();
	~cDevice();

private:
	ComPtr<ID3D11Device> device;
	ComPtr<ID3D11DeviceContext> deviceContext;
	//ID3D11Device* device;
	//ID3D11DeviceContext* deviceContext;

	ComPtr<IDXGISwapChain> swapChain;
	ComPtr<ID3D11RenderTargetView> renderTargetView;
	ComPtr<ID3D11DepthStencilView> depthStencilView;
	//IDXGISwapChain* swapChain;
	//ID3D11RenderTargetView* renderTargetView;
	//ID3D11DepthStencilView* depthStencilView;

public:
	void CreateDeviceAndSwapChain();
	void CreateBackBuffer();
	void SetRenderTarget();
	void Clear(Float4 color = Float4(0.1f, 0.1f, 0.125f, 1.0f));
	void Present();

	ID3D11Device* GetDevice() { return device.Get(); }
	ID3D11DeviceContext* GetDeviceContext() { return deviceContext.Get(); }
	IDXGISwapChain* GetSwapChain() { return swapChain.Get(); }
};

