# Yuuna 3D Game Engine
Opensource GameEngine for DX11
This project is opensource game engine used DirectX11.

# Used UI Lib
imgui
https://github.com/ocornut/imgui/tree/docking

# 3D File Import
Assimp
Fbxsdk

# Cuda Version
12.3

# Network
Asio 1.18.0

# Goal Node
/YuunaEngine
│── CMakeLists.txt
│── config.json     <-- 실행 파일 설정 (이름, 아이콘)
│── Engine.exe
│── tools/
│   ├── Encryptor   <-- 암호화 도구
│   ├── CMake       <-- 빌드 도구
│── assets/         <-- 원본 리소스 파일 (FBX, PNG, Shader 등)
│── Plugin/
│   ├── ex1/
│   │   ├── inc/
│   │   │   ├── ex1.h
│   │   ├── lib/
│   │   │   ├── x64/
│   │   │   │   ├── ex1.lib
│   │   ├── dll/
│   │   │   ├── x64/
│   │   │   │   ├── ex1.dll
│── icons/
│   ├── myicon.ico
