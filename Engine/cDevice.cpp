#include "framework.h"
#include "cDevice.h"

cDevice::cDevice()
{
	CreateDeviceAndSwapChain();
	CreateBackBuffer();
}

cDevice::~cDevice()
{
}

void cDevice::CreateDeviceAndSwapChain()
{
	UINT width = WIN_X;
	UINT height = WIN_Y;

	DXGI_SWAP_CHAIN_DESC desc = {};
	desc.BufferCount = 1;
	desc.BufferDesc.Width = width;
	desc.BufferDesc.Height = height;
	desc.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	desc.BufferDesc.RefreshRate.Numerator = 60;
	desc.BufferDesc.RefreshRate.Denominator = 1;
	desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
	desc.OutputWindow = g_hWnd;
	desc.SampleDesc.Count = 1;
	desc.SampleDesc.Quality = 0;
	desc.Windowed = !FULLSCREEN;

#ifdef _DEBUG
	V(D3D11CreateDeviceAndSwapChain(
		nullptr,
		D3D_DRIVER_TYPE_HARDWARE,
		0,
		D3D11_CREATE_DEVICE_DEBUG | D3D11_CREATE_DEVICE_BGRA_SUPPORT,
		nullptr,
		0,
		D3D11_SDK_VERSION,
		&desc,
		swapChain.GetAddressOf(),
		device.GetAddressOf(),
		nullptr,
		deviceContext.GetAddressOf()
	));
#else
	V(D3D11CreateDeviceAndSwapChain(
		nullptr,
		D3D_DRIVER_TYPE_HARDWARE,
		0,
		D3D11_CREATE_DEVICE_BGRA_SUPPORT,
		nullptr,
		0,
		D3D11_SDK_VERSION,
		&desc,
		swapChain.GetAddressOf(),
		device.GetAddressOf(),
		nullptr,
		deviceContext.GetAddressOf()
	));
#endif // _DEBUG

}

void cDevice::CreateBackBuffer()
{
	ID3D11Texture2D* backBuffer;

	V(swapChain->GetBuffer(0, __uuidof(ID3D11Texture2D), (void**)&backBuffer));
	V(device->CreateRenderTargetView(backBuffer, nullptr, renderTargetView.GetAddressOf()));
	backBuffer->Release();

	ID3D11Texture2D* depthBuffer;

	{
		D3D11_TEXTURE2D_DESC desc = {};
		desc.Width = WIN_X;
		desc.Height = WIN_Y;
		desc.MipLevels = 1;
		desc.ArraySize = 1;
		desc.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
		desc.SampleDesc.Count = 1;
		desc.SampleDesc.Quality = 0;
		desc.Usage = D3D11_USAGE_DEFAULT;
		desc.BindFlags = D3D11_BIND_DEPTH_STENCIL;

		V(device->CreateTexture2D(&desc, nullptr, &depthBuffer));
	}

	{
		D3D11_DEPTH_STENCIL_VIEW_DESC desc = {};
		desc.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
		desc.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;

		V(device->CreateDepthStencilView(depthBuffer, &desc, depthStencilView.GetAddressOf()));
		depthBuffer->Release();
	}
}

void cDevice::SetRenderTarget()
{
	deviceContext->OMSetRenderTargets(1, renderTargetView.GetAddressOf(), depthStencilView.Get());
}

void cDevice::Clear(Float4 color)
{
	deviceContext->ClearRenderTargetView(renderTargetView.Get(), (float*)&color);
	deviceContext->ClearDepthStencilView(depthStencilView.Get(), D3D11_CLEAR_DEPTH, 1.0f, 0);
}

void cDevice::Present()
{
	swapChain->Present(0, 0);
}
