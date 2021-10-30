#include <stdio.h>
#include <cuda.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#define DEFAULT_NUM_ELEMENTS 1 << 20
#define DEFAULT_BLOCK_SIZE 256

void printUsage(char *argv[]) {
    printf("Usage: %s <num_elements> <:OPTIONAL: threads_per_block>", argv[0]);
}

__global__ void cudaAdd(int *a, int *b, int *c, int maxIndex) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < maxIndex) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void cudaSub(int *a, int *b, int *c, int maxIndex) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < maxIndex) {
        c[idx] = a[idx] - b[idx];
    }
}

__global__ void cudaMult(int *a, int *b, int *c, int maxIndex) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < maxIndex) {
        c[idx] = a[idx] * b[idx];
    }
}

__global__ void cudaMod(int *a, int *b, int *c, int maxIndex) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < maxIndex) {
        c[idx] = a[idx] % b[idx];
    }
}

float cudaOps(int *a, int *b, int totalElements, int blockSize) {
    int numBlocks = (totalElements + blockSize - 1) / blockSize;

    // allocate device memory
    int *deviceA;
    int *deviceB;
    int *deviceC;
    cudaMalloc(&deviceA, totalElements * sizeof(*deviceA));
    cudaMalloc(&deviceB, totalElements * sizeof(*deviceB));
    cudaMalloc(&deviceC, totalElements * sizeof(*deviceC));

    // copy inputs to device
    cudaMemcpy(deviceA, a, totalElements * sizeof(*deviceA), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, b, totalElements * sizeof(*deviceB), cudaMemcpyHostToDevice);

    // setup timing
    float kernelTime;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // do each of the four operations with standard cuda
    cudaAdd<<<numBlocks, blockSize>>>(deviceA, deviceB, deviceC, totalElements);
    cudaSub<<<numBlocks, blockSize>>>(deviceA, deviceB, deviceC, totalElements);
    cudaMult<<<numBlocks, blockSize>>>(deviceA, deviceB, deviceC, totalElements);
    cudaMod<<<numBlocks, blockSize>>>(deviceA, deviceB, deviceC, totalElements);
    cudaDeviceSynchronize();

    // stop the clock
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&kernelTime, start, stop);

    // free resources
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);

    // return time
    return kernelTime;
}

float thrustOps(int *a, int *b, int totalElements) {
    // setup thrust vectors
    thrust::host_vector<int> hostA(totalElements);
    thrust::host_vector<int> hostB(totalElements);

    for (int i = 0; i < totalElements; i++) {
        hostA[i] = a[i];
        hostB[i] = b[i];
    }

    // copy inputs to device
    thrust::device_vector<int> deviceA = hostA;
    thrust::device_vector<int> deviceB = hostB;
    thrust::device_vector<int> deviceC(totalElements);

    // setup timing
    float kernelTime;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // do each of the four operations with thrust
    thrust::transform(deviceA.begin(), deviceA.end(), deviceB.begin(), deviceC.begin(), thrust::plus<int>());
    thrust::transform(deviceA.begin(), deviceA.end(), deviceB.begin(), deviceC.begin(), thrust::minus<int>());
    thrust::transform(deviceA.begin(), deviceA.end(), deviceB.begin(), deviceC.begin(), thrust::multiplies<int>());
    thrust::transform(deviceA.begin(), deviceA.end(), deviceB.begin(), deviceC.begin(), thrust::modulus<int>());

    // stop the clock
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&kernelTime, start, stop);

    // no need to free, thrust vectors will be destroyed when context switches

    // return time
    return kernelTime;
}

int main(int argc, char *argv[]) {
    // parse args
    if (argc > 3) {
        printUsage(argv);
    }

    int totalSize = DEFAULT_NUM_ELEMENTS;
    if (argc > 1) {
        totalSize = atoi(argv[1]);
    }

    int blockSize = DEFAULT_BLOCK_SIZE;
    if (argc > 2) {
        blockSize = atoi(argv[2]);
    }

    // generate data
    int *a = (int *) malloc(totalSize * sizeof(*a));
    int *b = (int *) malloc(totalSize * sizeof(*b));

    for (int i = 0; i < totalSize; i++) {
        a[i] = i;
        b[i] = 1 + (rand() % RAND_MAX);
    }

    // do the four operations each way
    float cudaTime = cudaOps(a, b, totalSize, blockSize);
    float thrustTime = thrustOps(a, b, totalSize);

    // print results
    printf("Total elements: %d -- Threads Per Block: %d\n", totalSize, blockSize);
    printf("Standard CUDA time... : %f (ms)\n", cudaTime);
    printf("Thrust time... : %f (ms)\n", thrustTime);

    // free resources
    free(a);
    free(b);

    return 0;
}