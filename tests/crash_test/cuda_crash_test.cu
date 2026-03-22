/**
 * CUDA Crash Test Program
 *
 * This program intentionally contains bugs for testing cuda-gdb-cli.
 * Each test function demonstrates a different type of CUDA error.
 *
 * Build: make
 * Run: ./cuda_crash_test <test_number>
 *
 * Test cases:
 *   1 - Illegal memory access (out-of-bounds global memory)
 *   2 - Shared memory race condition
 *   3 - Warp divergence issue
 *   4 - Null pointer dereference
 *   5 - Stack overflow
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)


// =============================================================================
// Test 1: Illegal Memory Access (Out-of-Bounds)
// =============================================================================

__global__ void illegal_memory_access_kernel(float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Bug: Writing beyond allocated memory
    // This will cause CUDA_EXCEPTION_LANE_ILLEGAL_ADDRESS
    if (idx == 0) {
        // Intentionally access beyond bounds
        output[size + 1000] = 42.0f;  // Out of bounds write
    }
}

void test_illegal_memory_access() {
    printf("=== Test 1: Illegal Memory Access ===\n");

    const int size = 1024;
    float* d_output;

    CUDA_CHECK(cudaMalloc(&d_output, size * sizeof(float)));

    dim3 grid(1, 1, 1);
    dim3 block(256, 1, 1);

    illegal_memory_access_kernel<<<grid, block>>>(d_output, size);

    // This will detect the error
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(d_output));
    printf("Test 1 completed without error (unexpected)\n");
}


// =============================================================================
// Test 2: Shared Memory Race Condition
// =============================================================================

__global__ void shared_memory_race_kernel(int* output) {
    __shared__ int shared_data[256];

    int tid = threadIdx.x;

    // Bug: No __syncthreads() between write and read
    // This causes a race condition
    shared_data[tid] = tid;

    // Missing __syncthreads() here!

    // Each thread reads from a different location
    int other_tid = (tid + 128) % 256;
    output[blockIdx.x * blockDim.x + tid] = shared_data[other_tid];
}

void test_shared_memory_race() {
    printf("=== Test 2: Shared Memory Race Condition ===\n");

    const int size = 256;
    int* d_output;
    int* h_output = (int*)malloc(size * sizeof(int));

    CUDA_CHECK(cudaMalloc(&d_output, size * sizeof(int)));

    shared_memory_race_kernel<<<1, 256>>>(d_output);

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_output, d_output, size * sizeof(int), cudaMemcpyDeviceToHost));

    printf("Output (may vary due to race): ");
    for (int i = 0; i < 10; i++) {
        printf("%d ", h_output[i]);
    }
    printf("...\n");

    CUDA_CHECK(cudaFree(d_output));
    free(h_output);
    printf("Test 2 completed\n");
}


// =============================================================================
// Test 3: Warp Divergence Issue
// =============================================================================

__global__ void warp_divergence_kernel(float* output, int* condition) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Bug: Divergent code path within a warp
    // Half the warp takes one path, half takes another
    if (idx % 2 == 0) {
        // Even threads: slow path
        for (int i = 0; i < 1000; i++) {
            output[idx] += 0.001f;
        }
    } else {
        // Odd threads: fast path
        output[idx] = 0.0f;
    }

    // This can cause performance issues and potentially incorrect results
    // if the divergent paths modify shared state incorrectly
}

void test_warp_divergence() {
    printf("=== Test 3: Warp Divergence ===\n");

    const int size = 256;
    float* d_output;
    int* d_condition;

    CUDA_CHECK(cudaMalloc(&d_output, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_condition, size * sizeof(int)));

    warp_divergence_kernel<<<1, 256>>>(d_output, d_condition);

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_condition));
    printf("Test 3 completed\n");
}


// =============================================================================
// Test 4: Null Pointer Dereference
// =============================================================================

__global__ void null_pointer_kernel(float* ptr) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Bug: Dereferencing a null pointer
    if (ptr == nullptr && idx == 0) {
        // This will crash
        ptr[idx] = 42.0f;
    }
}

void test_null_pointer() {
    printf("=== Test 4: Null Pointer Dereference ===\n");

    null_pointer_kernel<<<1, 256>>>(nullptr);

    // This will detect the error
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Expected error occurred: %s\n", cudaGetErrorString(err));
    } else {
        printf("Test 4 completed without error (unexpected)\n");
    }
}


// =============================================================================
// Test 5: Stack Overflow
// =============================================================================

__device__ int recursive_function(int n) {
    // Bug: Infinite recursion without proper base case
    if (n <= 0) {
        return 0;
    }
    return n + recursive_function(n - 1);
}

__global__ void stack_overflow_kernel(int* output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // This may cause stack overflow for large n
    output[idx] = recursive_function(10000);
}

void test_stack_overflow() {
    printf("=== Test 5: Stack Overflow ===\n");

    int* d_output;
    CUDA_CHECK(cudaMalloc(&d_output, 256 * sizeof(int)));

    stack_overflow_kernel<<<1, 256>>>(d_output);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Error occurred: %s\n", cudaGetErrorString(err));
    }

    CUDA_CHECK(cudaFree(d_output));
    printf("Test 5 completed\n");
}


// =============================================================================
// Test 6: Matrix Multiplication with Bug (for debugging demo)
// =============================================================================

#define TILE_SIZE 16

__global__ void matmul_kernel(float* A, float* B, float* C, int M, int N, int K) {
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tiles into shared memory
        int a_col = t * TILE_SIZE + threadIdx.x;
        int b_row = t * TILE_SIZE + threadIdx.y;

        if (row < M && a_col < K) {
            tile_A[threadIdx.y][threadIdx.x] = A[row * K + a_col];
        } else {
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (b_row < K && col < N) {
            tile_B[threadIdx.y][threadIdx.x] = B[b_row * N + col];
        } else {
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Compute partial sum
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        }

        __syncthreads();
    }

    // Bug: Off-by-one error in boundary check
    // Should be: if (row < M && col < N)
    if (row <= M && col <= N) {  // Bug: <= instead of <
        C[row * N + col] = sum;
    }
}

void test_matmul() {
    printf("=== Test 6: Matrix Multiplication (with bug) ===\n");

    const int M = 32, N = 32, K = 32;

    float *h_A = (float*)malloc(M * K * sizeof(float));
    float *h_B = (float*)malloc(K * N * sizeof(float));
    float *h_C = (float*)malloc(M * N * sizeof(float));

    // Initialize matrices
    for (int i = 0; i < M * K; i++) h_A[i] = 1.0f;
    for (int i = 0; i < K * N; i++) h_B[i] = 1.0f;

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice));

    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    dim3 block(TILE_SIZE, TILE_SIZE);

    matmul_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify result
    printf("Sample output C[0:4]: ");
    for (int i = 0; i < 4; i++) {
        printf("%.1f ", h_C[i]);
    }
    printf("\n");

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);
    printf("Test 6 completed\n");
}


// =============================================================================
// Main
// =============================================================================

void print_usage(const char* prog) {
    printf("Usage: %s <test_number>\n", prog);
    printf("\nAvailable tests:\n");
    printf("  1 - Illegal memory access (out-of-bounds)\n");
    printf("  2 - Shared memory race condition\n");
    printf("  3 - Warp divergence issue\n");
    printf("  4 - Null pointer dereference\n");
    printf("  5 - Stack overflow\n");
    printf("  6 - Matrix multiplication with bug\n");
    printf("  all - Run all tests\n");
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    // Print device info
    int device;
    CUDA_CHECK(cudaGetDevice(&device));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    printf("CUDA Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("SMs: %d\n", prop.multiProcessorCount);
    printf("\n");

    if (strcmp(argv[1], "all") == 0) {
        test_illegal_memory_access();
        test_shared_memory_race();
        test_warp_divergence();
        test_null_pointer();
        test_stack_overflow();
        test_matmul();
    } else {
        int test = atoi(argv[1]);
        switch (test) {
            case 1: test_illegal_memory_access(); break;
            case 2: test_shared_memory_race(); break;
            case 3: test_warp_divergence(); break;
            case 4: test_null_pointer(); break;
            case 5: test_stack_overflow(); break;
            case 6: test_matmul(); break;
            default:
                printf("Unknown test: %d\n", test);
                print_usage(argv[0]);
                return 1;
        }
    }

    return 0;
}