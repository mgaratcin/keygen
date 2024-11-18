#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <inttypes.h> // Include for portable format specifiers
#include "GPU/GPUEngine.h"
#include "SECP256k1.h"
#include "hash/sha256.h"
#include "hash/ripemd160.h"
#include "Int.h"
#include "Point.h"

#define START_KEY 0x4000000000000000ULL
#define END_KEY   0x4FFFFFFFFFFFFFFFULL
#define BILLION   1000000000ULL
#define STEP_SIZE 1024

// Target RIPEMD-160 hash
const std::string TARGET_HASH = "739437bb3dd6d1983e66629c5f08c70e52769371";

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

    // Convert target hash to binary format
    uint8_t targetHash[20];
    for (size_t i = 0; i < TARGET_HASH.size(); i += 2) {
        targetHash[i / 2] = static_cast<uint8_t>(strtol(TARGET_HASH.substr(i, 2).c_str(), nullptr, 16));
    }

    // Debugging: Log the target hash
    std::cout << "Target Hash160: ";
    for (int i = 0; i < 20; ++i) {
        printf("%02x", targetHash[i]);
    }
    std::cout << std::endl;

    // Initialize SECP256k1 curve
    Secp256K1 secp;
    secp.Init();

    // Start the GPU search
    for (uint64_t privateKeyStart = START_KEY; privateKeyStart <= END_KEY; privateKeyStart += BILLION) {
        // Launch the GPU kernel
        std::vector<ITEM> results;
        gpuEngine.Launch(results, false);
	//
        // Log results returned from GPU for debugging
        //if (results.empty()) {
        //    std::cout << "No results returned from GPU for this range." << std::endl;
        //    continue;
        //} else {
        //    std::cout << "Results returned from GPU: " << results.size() << std::endl;
        //}

        // Process all results returned from the GPU
        for (const auto& item : results) {
            uint64_t privateKeyValue = privateKeyStart + (item.thId * STEP_SIZE) + item.incr;

            Int privateKey(privateKeyValue);

            // Compute public key
            Point publicKey = secp.ComputePublicKey(&privateKey);

            // Serialize public key (compressed format)
            std::vector<uint8_t> pubKeyBytes(33);
            pubKeyBytes[0] = publicKey.y.IsEven() ? 0x02 : 0x03;
            publicKey.x.Get32Bytes(&pubKeyBytes[1]);

            // Compute hashes
            uint8_t sha256Hash[32];
            sha256(pubKeyBytes.data(), static_cast<int>(pubKeyBytes.size()), sha256Hash);

            uint8_t ripemd160Hash[20];
            ripemd160(sha256Hash, 32, ripemd160Hash);

            // Log directly from GPU computation
            // Fixed escape sequences in printf statements
            printf("GPU Debug - Private Key: %" PRIx64 "\n", privateKeyValue);

            printf("GPU Debug - Public Key: ");
            for (size_t i = 0; i < pubKeyBytes.size(); ++i) {
                printf("%02x", pubKeyBytes[i]);
            }
            printf("\n");

            printf("GPU Debug - Generated Hash160: ");
            for (int i = 0; i < 20; ++i) {
                printf("%02x", ripemd160Hash[i]);
            }
            printf("\n");

            // Check for a match
            bool isMatch = true;
            for (int i = 0; i < 20; ++i) {
                if (ripemd160Hash[i] != targetHash[i]) {
                    isMatch = false;
                    break;
                }
            }

            if (isMatch) {
                printf("Matching Key Found!\n");
                printf("Private Key: %" PRIx64 "\n", privateKeyValue);
                return 0; // Exit on first match
            }
        }
    }

    printf("Processing complete. No matching key found.\n");
    return 0;
}
