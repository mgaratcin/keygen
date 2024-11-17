#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <algorithm>
#include <sstream>
#include "GPU/GPUEngine.h"
#include "Secp256K1.h"
#include "hash/sha256.h"
#include "hash/ripemd160.h"
#include "Int.h"
#include "Point.h"

#define START_KEY 0x4000000000000000ULL
#define END_KEY   0x6FFFFFFFFFFFFFFFULL
#define BILLION   1000000000ULL
#define STEP_SIZE 1024 // Match kernel's step size

// Target RIPEMD-160 hash
const std::string TARGET_HASH = "739437bb3dd6d1983e66629c5f08c70e52769371";

// Helper function to convert binary data to a lowercase hex string
std::string hashToHex(const uint8_t* data, size_t len) {
    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    for (size_t i = 0; i < len; ++i) {
        oss << std::setw(2) << std::setw(2) << static_cast<int>(data[i]);
    }
    return oss.str();
}

int main() {
    // GPU engine parameters
    const int gpuId = 0;
    const int nbThreadGroup = 1024; // Number of thread groups
    const int nbThreadPerGroup = 128; // Threads per group
    const uint32_t maxFound = 262144; // Max results
    const bool rekey = false;

    // Create GPU engine
    GPUEngine gpuEngine(nbThreadGroup, nbThreadPerGroup, gpuId, maxFound, rekey);

    // Set search mode and type
    gpuEngine.SetSearchMode(SEARCH_COMPRESSED);
    gpuEngine.SetSearchType(P2PKH);

    // Print GPU info
    GPUEngine::PrintCudaInfo();

    // Convert target hash to binary format
    uint8_t targetHash[20];
    for (size_t i = 0; i < TARGET_HASH.size(); i += 2) {
        targetHash[i / 2] = static_cast<uint8_t>(strtol(TARGET_HASH.substr(i, 2).c_str(), nullptr, 16));
    }

    // Initialize prefix vector (only using the first few bytes for prefix matching)
    std::vector<prefix_t> prefixes;
    prefix_t prefix;
    memcpy(&prefix, targetHash, sizeof(prefix_t));
    prefixes.push_back(prefix);

    gpuEngine.SetPrefix(prefixes);

    uint64_t totalKeysProcessed = 0;

    // Initialize SECP256k1 curve
    Secp256K1 secp;
    secp.Init();

    // Start the GPU search
    for (uint64_t privateKeyStart = START_KEY; privateKeyStart <= END_KEY; privateKeyStart += BILLION) {
        std::vector<ITEM> results;

        // Launch GPU kernel to search within the current key range
        gpuEngine.Launch(results, false);

        // Check results
        for (const auto& item : results) {
            // Compute the private key based on the thread ID and increment
            uint64_t privateKeyValue = privateKeyStart + (item.thId * STEP_SIZE) + item.incr;
            Int privateKey(privateKeyValue);

            // Compute public key
            Point publicKey = secp.ComputePublicKey(&privateKey);

            // Serialize the public key
            std::vector<uint8_t> pubKeyBytes;
            bool compressed = item.mode;
            if (compressed) {
                // For compressed public key
                pubKeyBytes.resize(33);
                pubKeyBytes[0] = publicKey.y.IsEven() ? 0x02 : 0x03;
                publicKey.x.Get32Bytes(&pubKeyBytes[1]);
            } else {
                // For uncompressed public key
                pubKeyBytes.resize(65);
                pubKeyBytes[0] = 0x04;
                publicKey.x.Get32Bytes(&pubKeyBytes[1]);
                publicKey.y.Get32Bytes(&pubKeyBytes[33]);
            }

            // Compute SHA256 hash of the public key
            uint8_t sha256Hash[32];
            sha256(pubKeyBytes.data(), static_cast<int>(pubKeyBytes.size()), sha256Hash);

            // Compute RIPEMD160 hash of the SHA256 hash
            uint8_t ripemd160Hash[20];
            ripemd160(sha256Hash, 32, ripemd160Hash);

            // Compare with target hash
            if (memcmp(ripemd160Hash, targetHash, 20) == 0) {
                // Match found
                std::cout << "Match Found!" << std::endl;
                std::cout << "Thread ID: " << item.thId << std::endl;
                std::cout << "Incr: " << item.incr << std::endl;
                std::cout << "Endo: " << item.endo << std::endl;
                std::cout << "Hash160: " << hashToHex(ripemd160Hash, 20) << std::endl;
                std::cout << "Mode: " << (compressed ? "Compressed" : "Uncompressed") << std::endl;
                std::cout << "Private Key: " << std::hex << privateKeyValue << std::endl;
                std::cout << "Public Key: " << hashToHex(pubKeyBytes.data(), pubKeyBytes.size()) << std::endl;

                return 0;
            }
        }

        totalKeysProcessed += BILLION;

        // Log progress every 100 billion keys
        if (totalKeysProcessed % (100ULL * BILLION) == 0) {
            std::cout << "Processed " << totalKeysProcessed << " keys so far..." << std::endl;
        }
    }

    std::cout << "Processing complete. No matching key found." << std::endl;
    return 0;
}
