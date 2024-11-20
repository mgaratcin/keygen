#include <stdio.h>
#include <stdint.h>
#include <inttypes.h> // For PRIu64 and PRIx64
#include <cuda.h>

// Define GRP_SIZE before including GPUMath.h
#define GRP_SIZE 64

#include "GPUMath.h" // Ensure this header does not use __uint128_t

// Structure to represent a 256-bit integer (with an extra limb for GPUMath)
typedef struct {
    uint64_t limb[5]; // limb[0] is least significant
} uint256;

// Structure to represent a point on the elliptic curve
typedef struct {
    uint256 x;
    uint256 y;
} Point;

// Function Prototypes
__device__ void point_double(Point* R, const Point* P);
__device__ void point_add(Point* R, const Point* P, const Point* Q);
__device__ bool uint256_eq(const uint256 a, const uint256 b);
__device__ bool uint256_is_zero(const uint256 a);
__device__ void uint256_copy(uint256* dest, const uint256* src);
__device__ void uint256_add_mod(uint256* r, const uint256* a, const uint256* b);
__device__ void uint256_sub_mod(uint256* r, const uint256* a, const uint256* b);
__device__ void uint256_mult_mod(uint256* r, const uint256* a, const uint256* b);
__device__ void uint256_inv_mod(uint256* inv, const uint256* a);
__device__ void scalar_multiplication(Point* R, const uint256* k, const Point* G_point);

// Helper Functions for Addition and Subtraction with Carry/Borrow
__device__ void add_with_carry(uint64_t a, uint64_t b, uint64_t carry_in, uint64_t* sum, uint64_t* carry_out) {
    uint64_t temp_sum = a + b;
    *carry_out = (temp_sum < a) ? 1 : 0;
    temp_sum += carry_in;
    if (temp_sum < carry_in) {
        *carry_out += 1;
    }
    *sum = temp_sum;
}

__device__ void sub_with_borrow(uint64_t a, uint64_t b, uint64_t borrow_in, uint64_t* diff, uint64_t* borrow_out) {
    uint64_t temp_diff = a - b - borrow_in;
    *borrow_out = (a < (b + borrow_in)) ? 1 : 0;
    *diff = temp_diff;
}

// Define the prime field P for secp256k1 as FIELD_P to avoid naming conflicts
__device__ __constant__ uint256 FIELD_P = {
    .limb = {
        0xFFFFFFFEFFFFFC2FULL, // limb[0]
        0xFFFFFFFFFFFFFFFFULL, // limb[1]
        0xFFFFFFFFFFFFFFFFULL, // limb[2]
        0xFFFFFFFFFFFFFFFFULL, // limb[3]
        0x0ULL                 // limb[4]
    }
};

// Define the generator point G for secp256k1 with corrected limb assignments
__device__ __constant__ Point G = {
    .x = {
        .limb = {
            0x59F2815B16F81798ULL, // limb[0] (Least Significant)
            0x029BFCDB2DCE28D9ULL, // limb[1]
            0x55A06295CE870B07ULL, // limb[2]
            0x79BE667EF9DCBBACULL, // limb[3] (Most Significant)
            0x0ULL                  // limb[4]
        }
    },
    .y = {
        .limb = {
            0x9C47D08FFB10D4B8ULL, // limb[0] (Least Significant)
            0xFD17B448A6855419ULL, // limb[1]
            0x5DA4FBFC0E1108A8ULL, // limb[2]
            0x483ADA7726A3C465ULL, // limb[3] (Most Significant)
            0x0ULL                  // limb[4]
        }
    }
};

// Function to compare two uint256 numbers
__device__ bool uint256_eq(const uint256 a, const uint256 b) {
    return (a.limb[0] == b.limb[0]) &&
           (a.limb[1] == b.limb[1]) &&
           (a.limb[2] == b.limb[2]) &&
           (a.limb[3] == b.limb[3]) &&
           (a.limb[4] == b.limb[4]);
}

// Function to check if uint256 is zero
__device__ bool uint256_is_zero(const uint256 a) {
    return (a.limb[0] | a.limb[1] | a.limb[2] | a.limb[3] | a.limb[4]) == 0;
}

// Function to copy uint256
__device__ void uint256_copy(uint256* dest, const uint256* src) {
    dest->limb[0] = src->limb[0];
    dest->limb[1] = src->limb[1];
    dest->limb[2] = src->limb[2];
    dest->limb[3] = src->limb[3];
    dest->limb[4] = src->limb[4];
}

// Function to perform modular addition: (a + b) mod FIELD_P
__device__ void uint256_add_mod(uint256* r, const uint256* a, const uint256* b) {
    uint64_t temp[5];
    uint64_t carry = 0;

    for(int i = 0; i < 4; i++) {
        add_with_carry(a->limb[i], b->limb[i], carry, &temp[i], &carry);
    }

    // Since limb[4] is always 0 in FIELD_P, we can add limb[4] without carry
    add_with_carry(a->limb[4], b->limb[4], carry, &temp[4], &carry);

    // Check if temp >= FIELD_P
    bool needs_sub = false;
    if (temp[3] > FIELD_P.limb[3] ||
        (temp[3] == FIELD_P.limb[3] && temp[2] > FIELD_P.limb[2]) ||
        (temp[3] == FIELD_P.limb[3] && temp[2] == FIELD_P.limb[2] && temp[1] > FIELD_P.limb[1]) ||
        (temp[3] == FIELD_P.limb[3] && temp[2] == FIELD_P.limb[2] && temp[1] == FIELD_P.limb[1] && temp[0] >= FIELD_P.limb[0])) {
        needs_sub = true;
    }

    if (needs_sub) {
        // temp = temp - FIELD_P
        uint64_t borrow = 0;
        sub_with_borrow(temp[0], FIELD_P.limb[0], borrow, &temp[0], &borrow);
        sub_with_borrow(temp[1], FIELD_P.limb[1], borrow, &temp[1], &borrow);
        sub_with_borrow(temp[2], FIELD_P.limb[2], borrow, &temp[2], &borrow);
        sub_with_borrow(temp[3], FIELD_P.limb[3], borrow, &temp[3], &borrow);
        sub_with_borrow(temp[4], FIELD_P.limb[4], borrow, &temp[4], &borrow);
        // limb[4] is zero, no need to handle further
    }

    // Assign result
    r->limb[0] = temp[0];
    r->limb[1] = temp[1];
    r->limb[2] = temp[2];
    r->limb[3] = temp[3];
    r->limb[4] = temp[4];
}

// Function to perform modular subtraction: (a - b) mod FIELD_P
__device__ void uint256_sub_mod(uint256* r, const uint256* a, const uint256* b) {
    uint64_t temp[5];
    uint64_t borrow = 0;

    for(int i = 0; i < 4; i++) {
        sub_with_borrow(a->limb[i], b->limb[i], borrow, &temp[i], &borrow);
    }

    // Subtract limb[4] with borrow
    sub_with_borrow(a->limb[4], b->limb[4], borrow, &temp[4], &borrow);

    // If borrow occurred, add FIELD_P
    if (borrow) {
        uint64_t carry = 0;
        for(int i = 0; i < 4; i++) {
            add_with_carry(temp[i], FIELD_P.limb[i], carry, &temp[i], &carry);
        }
        add_with_carry(temp[4], FIELD_P.limb[4], carry, &temp[4], &carry);
    }

    // Assign result
    r->limb[0] = temp[0];
    r->limb[1] = temp[1];
    r->limb[2] = temp[2];
    r->limb[3] = temp[3];
    r->limb[4] = temp[4];
}

// Function to perform modular multiplication: (a * b) mod FIELD_P
__device__ void uint256_mult_mod(uint256* r, const uint256* a, const uint256* b) {
    // Implementing full 256-bit multiplication and modular reduction using GPUMath
    uint64_t a_arr[5] = {a->limb[0], a->limb[1], a->limb[2], a->limb[3], 0};
    uint64_t b_arr[5] = {b->limb[0], b->limb[1], b->limb[2], b->limb[3], 0};
    uint64_t r_arr[4] = {0, 0, 0, 0}; // Store result as 4 limbs

    // Call _ModMult, assuming it writes to r_arr
    _ModMult(r_arr, a_arr, b_arr); // Ensure _ModMult is correctly implemented without __uint128_t

    // Assign result
    r->limb[0] = r_arr[0];
    r->limb[1] = r_arr[1];
    r->limb[2] = r_arr[2];
    r->limb[3] = r_arr[3];
    r->limb[4] = 0; // limb[4] not used
}

// Function to compute modular inverse: inv = a^{-1} mod FIELD_P
__device__ void uint256_inv_mod(uint256* inv, const uint256* a) {
    // Copy a to a temporary array of 5 limbs
    uint64_t a_arr[5];
    for(int i = 0; i < 5; i++) {
        a_arr[i] = a->limb[i];
    }
    _ModInv(a_arr); // Compute a^{-1} mod FIELD_P, result in a_arr

    // Assign result to inv
    inv->limb[0] = a_arr[0];
    inv->limb[1] = a_arr[1];
    inv->limb[2] = a_arr[2];
    inv->limb[3] = a_arr[3];
    inv->limb[4] = a_arr[4];
}

// Function to perform point doubling: R = 2P
__device__ void point_double(Point* R, const Point* P) {
    // Check if P is the point at infinity
    bool P_is_inf = uint256_is_zero(P->x) && uint256_is_zero(P->y);
    if (P_is_inf) {
        uint256_copy(&R->x, &P->x);
        uint256_copy(&R->y, &P->y);
        return;
    }

    // Check if P.y is zero, which would mean the tangent is vertical and R is point at infinity
    bool P_y_zero = uint256_is_zero(P->y);
    if (P_y_zero) {
        // R is point at infinity
        R->x.limb[0] = R->x.limb[1] = R->x.limb[2] = R->x.limb[3] = 0;
        R->x.limb[4] = 0;
        R->y.limb[0] = R->y.limb[1] = R->y.limb[2] = R->y.limb[3] = 0;
        R->y.limb[4] = 0;
        return;
    }

    // Calculate lambda = (3 * P.x^2) / (2 * P.y) mod FIELD_P
    uint256 numerator, denominator, lambda;
    // Compute 3 * P.x^2
    uint256_mult_mod(&numerator, &P->x, &P->x); // P.x^2 mod FIELD_P
    // Multiply numerator by 3: numerator = 3 * P.x^2 mod FIELD_P
    uint256 three = { .limb = {3ULL, 0, 0, 0, 0} };
    uint256_mult_mod(&numerator, &numerator, &three); // numerator = 3 * P.x^2 mod FIELD_P

    // Compute 2 * P.y: denominator = 2 * P.y mod FIELD_P
    uint256 two = { .limb = {2ULL, 0, 0, 0, 0} };
    uint256_mult_mod(&denominator, &two, &P->y); // denominator = 2 * P.y mod FIELD_P

    // Compute denominator inverse: denominator_inv = denominator^{-1} mod FIELD_P
    uint256 denominator_inv;
    uint256_inv_mod(&denominator_inv, &denominator); // denominator_inv = denominator^{-1} mod FIELD_P

    // Compute lambda = numerator * denominator_inv mod FIELD_P
    uint256_mult_mod(&lambda, &numerator, &denominator_inv); // lambda = (3*P.x^2)/(2*P.y) mod FIELD_P

    // Calculate x3 = lambda^2 - 2 * P.x mod FIELD_P
    uint256 lambda_sq, temp;
    uint256_mult_mod(&lambda_sq, &lambda, &lambda); // lambda^2 mod FIELD_P
    uint256_mult_mod(&temp, &two, &P->x); // 2 * P.x mod FIELD_P
    uint256_sub_mod(&R->x, &lambda_sq, &temp); // R.x = lambda^2 - 2 * P.x mod FIELD_P

    // Calculate y3 = lambda * (P.x - R.x) - P.y mod FIELD_P
    uint256 temp2, temp3;
    uint256_sub_mod(&temp2, &P->x, &R->x); // temp2 = P.x - R.x mod FIELD_P
    uint256_mult_mod(&temp3, &lambda, &temp2); // temp3 = lambda * (P.x - R.x) mod FIELD_P
    uint256_sub_mod(&R->y, &temp3, &P->y); // R.y = lambda * (P.x - R.x) - P.y mod FIELD_P
}

// Function to perform point addition: R = P + Q
__device__ void point_add(Point* R, const Point* P, const Point* Q) {
    // Check if P is the point at infinity
    bool P_is_inf = uint256_is_zero(P->x) && uint256_is_zero(P->y);
    if (P_is_inf) {
        uint256_copy(&R->x, &Q->x);
        uint256_copy(&R->y, &Q->y);
        return;
    }

    // Check if Q is the point at infinity
    bool Q_is_inf = uint256_is_zero(Q->x) && uint256_is_zero(Q->y);
    if (Q_is_inf) {
        uint256_copy(&R->x, &P->x);
        uint256_copy(&R->y, &P->y);
        return;
    }

    // Check if P == Q, if so use point_double
    bool P_eq_Q = uint256_eq(P->x, Q->x) && uint256_eq(P->y, Q->y);
    if (P_eq_Q) {
        point_double(R, P);
        return;
    }

    // Check if P.x == Q.x and P.y == -Q.y, then R is point at infinity
    bool P_x_eq_Q_x = uint256_eq(P->x, Q->x);
    bool P_y_eq_neg_Q_y = false;
    // To check if P.y = -Q.y mod FIELD_P, compute Q.y + P.y and see if it equals FIELD_P
    uint256 sum_y;
    uint256_add_mod(&sum_y, &Q->y, &P->y);
    if (uint256_eq(sum_y, FIELD_P)) {
        P_y_eq_neg_Q_y = true;
    }

    if (P_x_eq_Q_x && P_y_eq_neg_Q_y) {
        // R is point at infinity
        R->x.limb[0] = R->x.limb[1] = R->x.limb[2] = R->x.limb[3] = 0;
        R->x.limb[4] = 0;
        R->y.limb[0] = R->y.limb[1] = R->y.limb[2] = R->y.limb[3] = 0;
        R->y.limb[4] = 0;
        return;
    }

    // Calculate lambda = (Q.y - P.y) / (Q.x - P.x) mod FIELD_P
    uint256 numerator, denominator, lambda;
    uint256_sub_mod(&numerator, &Q->y, &P->y); // Q.y - P.y mod FIELD_P
    uint256_sub_mod(&denominator, &Q->x, &P->x); // Q.x - P.x mod FIELD_P

    // Check if denominator is zero, which would imply vertical line and point at infinity
    if (uint256_is_zero(denominator)) {
        // R is point at infinity
        R->x.limb[0] = R->x.limb[1] = R->x.limb[2] = R->x.limb[3] = 0;
        R->x.limb[4] = 0;
        R->y.limb[0] = R->y.limb[1] = R->y.limb[2] = R->y.limb[3] = 0;
        R->y.limb[4] = 0;
        return;
    }

    // Compute denominator inverse: denominator_inv = denominator^{-1} mod FIELD_P
    uint256 denominator_inv;
    uint256_inv_mod(&denominator_inv, &denominator); // denominator_inv = denominator^{-1} mod FIELD_P

    // Compute lambda = numerator * denominator_inv mod FIELD_P
    uint256_mult_mod(&lambda, &numerator, &denominator_inv); // lambda = (Q.y - P.y) / (Q.x - P.x) mod FIELD_P

    // Calculate x3 = lambda^2 - P.x - Q.x mod FIELD_P
    uint256 lambda_sq, temp;
    uint256_mult_mod(&lambda_sq, &lambda, &lambda); // lambda^2 mod FIELD_P
    uint256_add_mod(&temp, &P->x, &Q->x); // temp = P.x + Q.x mod FIELD_P
    uint256_sub_mod(&R->x, &lambda_sq, &temp); // R.x = lambda^2 - (P.x + Q.x) mod FIELD_P

    // Calculate y3 = lambda * (P.x - R.x) - P.y mod FIELD_P
    uint256 temp2, temp3;
    uint256_sub_mod(&temp2, &P->x, &R->x); // temp2 = P.x - R.x mod FIELD_P
    uint256_mult_mod(&temp3, &lambda, &temp2); // temp3 = lambda * (P.x - R.x) mod FIELD_P
    uint256_sub_mod(&R->y, &temp3, &P->y); // R.y = lambda * (P.x - R.x) - P.y mod FIELD_P
}

// Function to perform scalar multiplication using double-and-add
__device__ void scalar_multiplication(Point* R, const uint256* k, const Point* G_point) {
    // Initialize R to the point at infinity
    R->x.limb[0] = R->x.limb[1] = R->x.limb[2] = R->x.limb[3] = 0;
    R->x.limb[4] = 0;
    R->y.limb[0] = R->y.limb[1] = R->y.limb[2] = R->y.limb[3] = 0;
    R->y.limb[4] = 0;

    Point Q;
    uint256_copy(&Q.x, &G_point->x);
    uint256_copy(&Q.y, &G_point->y);

    // Iterate over each bit of the scalar k from LSB to MSB
    for(int i = 0; i < 256; i++) { // Iterate over 256 bits corresponding to 4 limbs
        int limb = i / 64;
        int bit = i % 64;
        if (limb < 4 && (k->limb[limb] & ((uint64_t)1 << bit))) { // Ensure limb < 4
            point_add(R, R, &Q);
        }
        point_double(&Q, &Q);
    }
}

// CUDA Kernel to compute public key from private key
__global__ void compute_pub_key(const uint256* priv_key, Point* pub_key) {
    // Each thread computes one public key
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= 1) return; // Adjust if THREADS > 1

    // Load private key
    uint256 k;
    uint256_copy(&k, &priv_key[idx]);

    // Compute pub_key = k * G
    scalar_multiplication(&pub_key[idx], &k, &G);
}

// Host function to print a uint256 number
void print_uint256(const uint256 a) {
    // Print limbs in big-endian order using PRIx64
    printf("0x%016" PRIx64 "%016" PRIx64 "%016" PRIx64 "%016" PRIx64 "\n",
        a.limb[3],
        a.limb[2],
        a.limb[1],
        a.limb[0]);
}

// Host main function
int main() {
    // Define a sample private key (256-bit)
    uint256 h_priv_key;
    h_priv_key.limb[0] = 0x0000000000000001ULL; // Example private key (very small for demonstration)
    h_priv_key.limb[1] = 0x0000000000000000ULL;
    h_priv_key.limb[2] = 0x0000000000000000ULL;
    h_priv_key.limb[3] = 0x0000000000000000ULL;
    h_priv_key.limb[4] = 0x0ULL;

    // Allocate memory on device
    uint256* d_priv_key;
    Point* d_pub_key;
    cudaError_t err;

    err = cudaMalloc((void**)&d_priv_key, sizeof(uint256));
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA malloc failed for priv_key: %s\n", cudaGetErrorString(err));
        return 1;
    }

    err = cudaMalloc((void**)&d_pub_key, sizeof(Point));
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA malloc failed for pub_key: %s\n", cudaGetErrorString(err));
        cudaFree(d_priv_key);
        return 1;
    }

    // Copy private key to device
    err = cudaMemcpy(d_priv_key, &h_priv_key, sizeof(uint256), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA memcpy failed for priv_key: %s\n", cudaGetErrorString(err));
        cudaFree(d_priv_key);
        cudaFree(d_pub_key);
        return 1;
    }

    // Launch kernel with 1 block and 1 thread
    compute_pub_key<<<1, 1>>>(d_priv_key, d_pub_key);

    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_priv_key);
        cudaFree(d_pub_key);
        return 1;
    }

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Allocate memory for public key on host
    Point h_pub_key;
    err = cudaMemcpy(&h_pub_key, d_pub_key, sizeof(Point), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA memcpy failed for pub_key: %s\n", cudaGetErrorString(err));
        cudaFree(d_priv_key);
        cudaFree(d_pub_key);
        return 1;
    }

    // Print the private key
    printf("Private Key:\n");
    print_uint256(h_priv_key);

    // Print the public key
    printf("Public Key:\n");
    printf("X: ");
    print_uint256(h_pub_key.x);
    printf("Y: ");
    print_uint256(h_pub_key.y);

    // Free device memory
    cudaFree(d_priv_key);
    cudaFree(d_pub_key);

    return 0;
}
