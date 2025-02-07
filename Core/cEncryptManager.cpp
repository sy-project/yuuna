#include "pch.h"
#include <openssl/aes.h>
#include <openssl/rand.h>

#pragma comment(lib, "libssl.lib")
#pragma comment(lib, "libcrypto.lib")

void cEncryptManager::Encrypt()
{
}

void cEncryptManager::Decrypt()
{
}

std::string cEncryptManager::EncryptData(std::string _data)
{
	std::string temp = "";
	for (int i = 0; i < _data.length(); i++)
	{
		temp += (_data.at(i) ^ (char)KEY);
	}
	return temp;
}

std::string cEncryptManager::DecryptData(std::string _data)
{
	std::string temp = "";
	for (int i = 0; i < _data.length(); i++)
	{
		temp += (_data.at(i) ^ (char)KEY);
	}
	return temp;
}

void cEncryptManager::EncryptResourceFile(const std::string& inputPath, const std::string& outputPath)
{
    // 1. 원본 파일 읽기
    std::ifstream inFile(inputPath, std::ios::binary);
    if (!inFile) {
        std::cerr << "Failed to open file: " << inputPath << std::endl;
        return;
    }

    std::vector<char> data((std::istreambuf_iterator<char>(inFile)), std::istreambuf_iterator<char>());
    inFile.close();

    if (data.empty()) {
        std::cerr << "Input file is empty: " << inputPath << std::endl;
        return;
    }

    // 2. OpenSSL 암호화 컨텍스트 생성
    EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
    if (!ctx) {
        std::cerr << "Failed to create OpenSSL cipher context" << std::endl;
        return;
    }

    if (EVP_EncryptInit_ex(ctx, EVP_aes_128_cbc(), nullptr, (unsigned char*)KEY, (unsigned char*)IV) != 1) {
        std::cerr << "Failed to initialize AES encryption" << std::endl;
        EVP_CIPHER_CTX_free(ctx);
        return;
    }

    // 3. 암호화 실행
    std::vector<char> encryptedData(data.size() + AES_BLOCK_SIZE);  // 암호화된 데이터 버퍼
    int outLen1 = 0;
    if (EVP_EncryptUpdate(ctx, (unsigned char*)encryptedData.data(), &outLen1, (unsigned char*)data.data(), data.size()) != 1) {
        std::cerr << "Failed to encrypt data" << std::endl;
        EVP_CIPHER_CTX_free(ctx);
        return;
    }

    int outLen2 = 0;
    if (EVP_EncryptFinal_ex(ctx, (unsigned char*)encryptedData.data() + outLen1, &outLen2) != 1) {
        std::cerr << "Failed to finalize encryption" << std::endl;
        EVP_CIPHER_CTX_free(ctx);
        return;
    }

    EVP_CIPHER_CTX_free(ctx);

    // 4. 암호화된 데이터 크기 조정
    encryptedData.resize(outLen1 + outLen2);

    // 5. 암호화된 데이터 파일에 저장
    std::ofstream outFile(outputPath, std::ios::binary);
    if (!outFile) {
        std::cerr << "Failed to open output file: " << outputPath << std::endl;
        return;
    }

    outFile.write(encryptedData.data(), encryptedData.size());
    outFile.close();

    std::cout << "Encrypted: " << inputPath << " -> " << outputPath << std::endl;
}

std::vector<char> cEncryptManager::DecryptResourceFile(const std::string& filePath)
{
    // 1. 암호화된 파일 읽기
    std::ifstream inFile(filePath, std::ios::binary);
    if (!inFile) {
        std::cerr << "Failed to open encrypted file: " << filePath << std::endl;
        return {};
    }

    std::vector<char> encryptedData((std::istreambuf_iterator<char>(inFile)), std::istreambuf_iterator<char>());
    inFile.close();

    if (encryptedData.empty()) {
        std::cerr << "Encrypted file is empty or invalid: " << filePath << std::endl;
        return {};
    }

    // 2. 복호화 준비
    EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
    if (!ctx) {
        std::cerr << "Failed to create OpenSSL cipher context" << std::endl;
        return {};
    }

    if (EVP_DecryptInit_ex(ctx, EVP_aes_128_cbc(), nullptr, (unsigned char*)KEY, (unsigned char*)IV) != 1) {
        std::cerr << "Failed to initialize AES decryption" << std::endl;
        EVP_CIPHER_CTX_free(ctx);
        return {};
    }

    // 3. 복호화 실행
    std::vector<char> decryptedData(encryptedData.size());
    int outLen1 = 0;
    if (EVP_DecryptUpdate(ctx, (unsigned char*)decryptedData.data(), &outLen1, (unsigned char*)encryptedData.data(), encryptedData.size()) != 1) {
        std::cerr << "Failed to decrypt data" << std::endl;
        EVP_CIPHER_CTX_free(ctx);
        return {};
    }

    int outLen2 = 0;
    if (EVP_DecryptFinal_ex(ctx, (unsigned char*)decryptedData.data() + outLen1, &outLen2) != 1) {
        std::cerr << "Failed to finalize decryption" << std::endl;
        EVP_CIPHER_CTX_free(ctx);
        return {};
    }

    EVP_CIPHER_CTX_free(ctx);

    // 4. 정확한 크기로 조정
    decryptedData.resize(outLen1 + outLen2);
    return decryptedData;
}

std::string cEncryptManager::EncryptNetwork(sPlayerDescription _data)
{
	std::string temp = "";
	return temp;
}

sPlayerDescription cEncryptManager::DecryptNetwork(std::string _data)
{
	return sPlayerDescription();
}

cEncryptManager::cEncryptManager()
{
}

cEncryptManager::~cEncryptManager()
{
}
