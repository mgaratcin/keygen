#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <sstream>
#include <algorithm> // For std::min
#include <getopt.h>  // For command-line argument parsing
#include <condition_variable> // For Semaphore
#include "GPU/GPUEngine.h"     // GPU Engine for computation
#include "SECP256k1.h"         // Ensure correct case for SECP256K1
#include "hash/sha256.h"       // Include your internal SHA256 header
#include "hash/ripemd160.h"    // Include your internal RIPEMD160 header
#include <cuda_runtime.h>      // CUDA runtime API

// Semaphore Implementation
class Semaphore {
public:
    Semaphore(int count_ = 0)
        : count(count_) {}

    void release() {
        std::unique_lock<std::mutex> lock(mtx);
        count++;
        cv.notify_one();
    }

    void acquire() {
        std::unique_lock<std::mutex> lock(mtx);
        while(count == 0){
            cv.wait(lock);
        }
        count--;
    }

private:
    std::mutex mtx;
    std::condition_variable cv;
    int count;
};

// Structure to hold configuration parameters
struct Config {
    uint64_t startKey = 0x1000000ULL;
    uint64_t endKey = 0x1ffffffffULL;
    std::string targetHash = "2f396b29b27324300d0c59b17c3abc1835bd3dbb";
    unsigned int numThreads = std::thread::hardware_concurrency();
    double memoryReservationFactor = 0.9; // Reserve 10% of free memory
};

// Function to parse command-line arguments
Config parseArguments(int argc, char* argv[]) {
    Config config;
    int opt;

    while ((opt = getopt(argc, argv, "s:e:t:m:")) != -1) {
        switch (opt) {
            case 's':
                config.startKey = std::stoull(optarg, nullptr, 16);
                break;
            case 'e':
                config.endKey = std::stoull(optarg, nullptr, 16);
                break;
            case 't':
                config.targetHash = std::string(optarg);
                break;
            case 'm':
                // Implement batch multiplier or other parameters as needed
                // For example, setting memory reservation factor
                config.memoryReservationFactor = std::stod(optarg);
                break;
            default:
                std::cerr << "Usage: " << argv[0] << " [-s startKey] [-e endKey] [-t targetHash] [-m memoryReservationFactor]" << std::endl;
                exit(EXIT_FAILURE);
        }
    }

    // Ensure numThreads is at least 1
    if (config.numThreads == 0) {
        config.numThreads = 4; // Default to 4 if unable to detect
    }

    return config;
}

// Function to calculate optimal batch size based on available GPU memory
uint64_t calculateOptimalBatchSize(int gpuId, unsigned int numThreads, double memoryFactor) {
    size_t freeMem = 0, totalMem = 0;
    cudaError_t cudaStatus = cudaSetDevice(gpuId);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA Error: Unable to set GPU device " << gpuId << std::endl;
        return 0;
    }

    cudaStatus = cudaMemGetInfo(&freeMem, &totalMem);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA Error: Unable to get memory info for GPU " << gpuId << std::endl;
        return 0;
    }

    // Calculate usable memory based on the reservation factor
    size_t usableMem = static_cast<size_t>(freeMem * memoryFactor);

    // Estimate memory per batch (adjust this based on actual memory usage per batch)
    // For demonstration, assume each batch consumes 1 MB
    size_t memPerBatch = 1 * 1024 * 1024; // 1 MB per batch

    // Calculate total possible batches
    size_t totalBatches = usableMem / memPerBatch;

    // Calculate batch size per thread
    uint64_t batchSize = totalBatches / numThreads;

    // Apply a minimum and maximum batch size
    const uint64_t MIN_BATCH_SIZE = 10;        // Minimum batch size
    const uint64_t MAX_BATCH_SIZE = 10000;     // Maximum batch size

    if (batchSize < MIN_BATCH_SIZE) batchSize = MIN_BATCH_SIZE;
    if (batchSize > MAX_BATCH_SIZE) batchSize = MAX_BATCH_SIZE;

    return batchSize;
}

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

// Helper function to calculate RIPEMD160(SHA256(pubkey))
std::string publicKeyToRIPEMD160(const Point& pubKey) {
    // Get the compressed public key as a hex string
    std::string compressedKeyHex = pointToCompressedHex(pubKey);
    
    // Convert hex string to bytes
    std::vector<uint8_t> compressedKeyBytes;
    compressedKeyBytes.reserve(compressedKeyHex.length() / 2);
    for (size_t i = 0; i < compressedKeyHex.length(); i += 2) {
        std::string byteString = compressedKeyHex.substr(i, 2);
        uint8_t byte = static_cast<uint8_t>(strtol(byteString.c_str(), nullptr, 16));
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
        oss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(ripemd160Hash[i]);
    }
    return oss.str();
}

// Global variables for synchronization
std::atomic<uint64_t> totalKeysProcessed(0);
std::atomic<bool> targetFound(false);
std::mutex coutMutex;

// Worker function for each thread
void worker(const std::string& TARGET_HASH, Semaphore& semaphore, std::atomic<uint64_t>& globalKeyCounter, uint64_t endKey, unsigned int numThreads, double memoryFactor) {
    // Acquire semaphore before initializing GPUEngine
    semaphore.acquire();

    // Instantiate and initialize SECP256K1 for this thread
    Secp256K1 secp;
    secp.Init();

    // Determine GPU ID based on thread ID and available GPUs
    int deviceCount = 0;
    cudaError_t cudaStatus = cudaGetDeviceCount(&deviceCount);
    if (cudaStatus != cudaSuccess || deviceCount == 0) {
        std::lock_guard<std::mutex> lock(coutMutex);
        std::cerr << "CUDA Error: Unable to get GPU count or no GPUs available." << std::endl;
        semaphore.release();
        return;
    }
    int gpuId = 0; // Default to GPU 0 or implement your own logic

    // Calculate optimal batch size
    uint64_t LARGE_BATCH_SIZE = calculateOptimalBatchSize(gpuId, numThreads, memoryFactor);
    if (LARGE_BATCH_SIZE == 0) {
        std::lock_guard<std::mutex> lock(coutMutex);
        std::cerr << "Error: Calculated batch size is 0. Exiting thread." << std::endl;
        semaphore.release();
        return;
    }

    const int nbThreadGroup = 10240;              // Number of thread groups
    const int nbThreadPerGroup = 128;             // Threads per group
    const uint64_t threadsPerIteration = static_cast<uint64_t>(nbThreadGroup) * nbThreadPerGroup;
    const uint32_t maxFound = 262144;             // Max items to be found
    const bool rekey = false;                     // No rekeying during the process

    // Create the GPU engine for this thread
    GPUEngine gpuEngine(nbThreadGroup, nbThreadPerGroup, gpuId, maxFound, rekey);

    // Set search mode and type (compressed keys)
    gpuEngine.SetSearchMode(SEARCH_COMPRESSED);
    gpuEngine.SetSearchType(P2PKH);

    // Allocate memory for batch processing
    std::vector<Int> privateKeysBatch;
    std::vector<Point> publicKeysBatch;
    privateKeysBatch.reserve(LARGE_BATCH_SIZE);
    publicKeysBatch.reserve(LARGE_BATCH_SIZE);

    while (!targetFound.load()) {
        // Fetch the next batch of keys atomically
        uint64_t startKey = globalKeyCounter.fetch_add(LARGE_BATCH_SIZE);
        if (startKey > endKey) break;

        uint64_t currentBatchSize = std::min(LARGE_BATCH_SIZE, endKey - startKey + 1);
        privateKeysBatch.clear();
        publicKeysBatch.clear();
        privateKeysBatch.reserve(currentBatchSize);
        publicKeysBatch.reserve(currentBatchSize);

        // Populate the batch
        for (uint64_t i = 0; i < currentBatchSize; ++i) {
            Int privKey(startKey + i); // Private key
            privateKeysBatch.push_back(privKey);
        }

        // Compute public keys for the entire batch
        for (auto& privKey : privateKeysBatch) { // Removed 'const' from the loop
            Point pubKey = secp.ComputePublicKey(&privKey); // Compute public key
            publicKeysBatch.push_back(pubKey);
        }

        // Process each public key in the batch
        for (size_t i = 0; i < publicKeysBatch.size(); ++i) {
            if (targetFound.load()) {
                break;
            }

            const Point& pubKey = publicKeysBatch[i];
            uint64_t currentPrivateKey = static_cast<uint64_t>(privateKeysBatch[i].bits64[0]);

            // Convert the public key to RIPEMD160 hash
            std::string rmd160Hash = publicKeyToRIPEMD160(pubKey);

            // Update the total keys processed
            uint64_t currentTotal = ++totalKeysProcessed;

            // Print progress at specified intervals
            if (currentTotal % TARGET_KEY_INTERVAL == 0) {
                std::lock_guard<std::mutex> lock(coutMutex);
                std::cout << "Target Key Interval Reached: " << currentTotal << " keys processed." << std::endl;
                std::cout << "Private Key: " << std::hex << currentPrivateKey << std::endl;
                std::cout << "Compressed Public Key: " << pointToCompressedHex(pubKey) << std::endl;
                std::cout << "RIPEMD160 Hash: " << rmd160Hash << std::endl;
            }

            // Check if the RIPEMD160 hash matches the target
            if (rmd160Hash == TARGET_HASH) {
                std::lock_guard<std::mutex> lock(coutMutex);
                std::cout << "Target RIPEMD160 Hash Found!" << std::endl;
                std::cout << "Private Key: " << std::hex << currentPrivateKey << std::endl;
                std::cout << "Compressed Public Key: " << pointToCompressedHex(pubKey) << std::endl;
                std::cout << "RIPEMD160 Hash: " << rmd160Hash << std::endl;
                targetFound.store(true);
                break; // Exit loop after finding the target key pair
            }
        }
    }

    // Release semaphore after GPU operations are complete
    semaphore.release();

    // Final status message for each thread
    {
        std::lock_guard<std::mutex> lock(coutMutex);
        std::cout << "Thread has completed its assigned range." << std::endl;
    }
}

int main(int argc, char* argv[]) {
    // Parse command-line arguments
    Config config = parseArguments(argc, argv);

    // Initialize semaphore with the number of concurrent GPUEngine instances allowed
    // For 12 GB VRAM and assuming each GPUEngine consumes ~1 GB, set to 12
    // Adjust this number based on actual GPU memory usage per GPUEngine
    const int MAX_CONCURRENT_GPU_ENGINES = 12;
    Semaphore semaphore(MAX_CONCURRENT_GPU_ENGINES);

    // Initialize a global key counter for sequential work allocation
    std::atomic<uint64_t> globalKeyCounter(config.startKey);

    // Print GPU info (only once)
    GPUEngine::PrintCudaInfo();

    std::cout << "Starting computation with " << config.numThreads << " threads." << std::endl;

    // Calculate the total range
    uint64_t totalRange = config.endKey - config.startKey + 1;

    // Create a vector to hold thread objects
    std::vector<std::thread> threads;

    for (unsigned int i = 0; i < config.numThreads; ++i) {
        // Launch a thread to handle sequential batches
        threads.emplace_back(worker, config.targetHash, std::ref(semaphore), std::ref(globalKeyCounter), config.endKey, config.numThreads, config.memoryReservationFactor);
    }

    // Wait for all threads to finish
    for (auto& th : threads) {
        if (th.joinable()) {
            th.join();
        }
    }

    if (!targetFound.load()) {
        std::cout << "Processing complete. Target RIPEMD160 Hash not found in the given range." << std::endl;
    }

    std::cout << "Total Keys Processed: " << totalKeysProcessed.load() << std::endl;
    std::cout << "Processing complete." << std::endl;
    return 0;
}
