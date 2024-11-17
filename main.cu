#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <algorithm>
#include "GPU/GPUEngine.h"
#include "SECP256k1.h"
#include "hash/sha256.h"
#include "hash/ripemd160.h"

#define START_KEY 0x4000000000000000ULL
#define END_KEY   0x6FFFFFFFFFFFFFFFULL
#define BILLION   1
#define STEP_SIZE 1024 // Match kernel's step size

// Target RIPEMD-160 hash
const std::string TARGET_HASH = "";

// Helper function to convert binary hash to a lowercase hex string
std::string hashToHex(const uint8_t* hash, size_t len) {
    std::ostringstream oss;
    for (size_t i = 0; i < len; ++i) {
        oss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(hash[i]);
    }
    return oss.str();
}

int main() {
    // GPU engine parameters
    const int gpuId = 0;
    const int nbThreadGroup = 1024; // Number of thread groups
    const int nbThreadPerGroup = 128; // Threads per group
    const uint32_t maxFound = 65536; // Max results
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

    // Initialize prefix vector
    std::vector<prefix_t> prefixes;
    prefixes.push_back(*(prefix_t*)targetHash); // Assuming the target hash is interpreted as a prefix.

    gpuEngine.SetPrefix(prefixes);

    uint64_t totalKeysProcessed = 0;

    // Start the GPU search
    for (uint64_t privateKeyStart = START_KEY; privateKeyStart <= END_KEY; privateKeyStart += BILLION) {
        std::vector<ITEM> results;

        // Launch GPU kernel to search within the current key range
        gpuEngine.Launch(results, false);

        // Check results
        for (const auto& item : results) {
            // Compute the private key based on the thread ID and increment
            uint64_t privateKey = privateKeyStart + (item.thId * STEP_SIZE) + item.incr;

            std::cout << "Match Found!" << std::endl;
            std::cout << "Thread ID: " << item.thId << std::endl;
            std::cout << "Incr: " << item.incr << std::endl;
            std::cout << "Endo: " << item.endo << std::endl;
            std::cout << "Hash: " << hashToHex(item.hash, 20) << std::endl;
            std::cout << "Mode: " << (item.mode ? "Compressed" : "Uncompressed") << std::endl;
            std::cout << "Private Key: " << std::hex << privateKey << std::endl;

            return 0;
        }

        totalKeysProcessed += BILLION;

        // Optional: Log progress
        if (totalKeysProcessed % (100 * BILLION) == 0) {
            std::cout << "Processed " << totalKeysProcessed << " keys so far..." << std::endl;
        }
    }

    std::cout << "Processing complete. No matching key found." << std::endl;
    return 0;
}
