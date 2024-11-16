#include <iostream>
#include <iomanip>
#include <chrono> // For timing
#include <vector>
#include "GPU/GPUEngine.h" // GPU Engine for computation
#include "SECP256k1.h"     // Ensure correct case for SECP256K1
#include "hash/sha256.h"   // Include your internal SHA256 header
#include "hash/ripemd160.h" // Include your internal RIPEMD160 header

#define START_KEY 0x4000000000000000ULL
#define END_KEY   0x7fffffffffffffffULL
#define BILLION   1000000000
#define TARGET_KEY_INTERVAL 100000000000ULL // Print every 100 billionth key 

// Helper function to convert Int to hex string
std::string intToHex(const Int& value) {
    std::ostringstream oss;
    for (int i = 3; i >= 0; --i) {
        oss << std::hex << std::setw(16) << std::setfill('0') << value.bits64[i];
    }
    return oss.str();
}

// Helper function to convert Point to compressed public key string
std::string pointToCompressedHex(const Point& point) {
    std::ostringstream oss;
    // Determine the prefix based on the parity of the y-coordinate
    std::string prefix = (point.y.bits64[0] & 1) == 0 ? "02" : "03";
    // Append the full x-coordinate
    oss << prefix << intToHex(point.x);
    return oss.str();
}

// Corrected helper function to calculate RIPEMD160(SHA256(pubkey))
std::string publicKeyToRIPEMD160(const Point& pubKey) {
    // Get the compressed public key as a hex string
    std::string compressedKeyHex = pointToCompressedHex(pubKey);
    
    // Convert hex string to bytes
    std::vector<uint8_t> compressedKeyBytes;
    for (size_t i = 0; i < compressedKeyHex.length(); i += 2) {
        std::string byteString = compressedKeyHex.substr(i, 2);
        uint8_t byte = (uint8_t) strtol(byteString.c_str(), nullptr, 16);
        compressedKeyBytes.push_back(byte);
    }

    // Compute SHA-256 hash of the compressed public key bytes
    uint8_t sha256Hash[32]; // SHA256 hash is 32 bytes
    sha256(compressedKeyBytes.data(), compressedKeyBytes.size(), sha256Hash);

    // Compute RIPEMD-160 hash of the SHA-256 hash
    uint8_t ripemd160Hash[20];
    ripemd160(sha256Hash, 32, ripemd160Hash);

    // Convert the RIPEMD-160 hash to a hex string
    std::ostringstream oss;
    for (int i = 0; i < 20; ++i) {
        oss << std::hex << std::setw(2) << std::setfill('0') << (int)ripemd160Hash[i];
    }
    return oss.str();
}

int main() {
    // Create an instance of Secp256K1
    Secp256K1 secp; // Correct casing matches the header file
    secp.Init();    // Initialize the curve parameters

    // Initialize GPU parameters
    const int gpuId = 0;                    // Use the first GPU
    const int nbThreadGroup = 1024;         // Number of thread groups
    const int nbThreadPerGroup = 128;       // Threads per group
    const uint64_t threadsPerIteration = nbThreadGroup * nbThreadPerGroup;
    const uint32_t maxFound = 1024;         // Max items to be found
    const bool rekey = false;               // No rekeying during the process

    // Create the GPU engine
    GPUEngine gpuEngine(nbThreadGroup, nbThreadPerGroup, gpuId, maxFound, rekey);

    // Set search mode and type (compressed keys)
    gpuEngine.SetSearchMode(SEARCH_COMPRESSED);
    gpuEngine.SetSearchType(P2PKH);

    // Print GPU info
    GPUEngine::PrintCudaInfo();

    uint64_t totalKeysProcessed = 0;

    // Loop through the range and check for 100 billionth key pairs
    for (uint64_t privateKey = START_KEY; privateKey <= END_KEY; privateKey += BILLION) {
        Int privKey(privateKey);                      // Private key
        Point pubKey = secp.ComputePublicKey(&privKey); // Compute public key

        // Aggregate total keys processed across all threads
        totalKeysProcessed += threadsPerIteration;

        // Check if this is the 100 billionth key pair
        if (totalKeysProcessed >= TARGET_KEY_INTERVAL) {
            std::cout << "100 Billionth Key Pair Found!" << std::endl;
            std::cout << "Private Key: 0x" << std::hex << privateKey << std::endl;
            std::cout << "Compressed Public Key: " << pointToCompressedHex(pubKey) << std::endl;

            // Convert the public key to RIPEMD160 hash and print it
            std::string rmd160Hash = publicKeyToRIPEMD160(pubKey);
            std::cout << "RIPEMD160 Hash: " << rmd160Hash << std::endl;

            break; // Exit loop after finding the target key pair
        }
    }

    std::cout << "Processing complete." << std::endl;
    return 0;
}
