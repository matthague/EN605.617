#include <stdio.h>
#include <assert.h>

#define DEFAULT_TOTAL_THREADS 256
#define DEFAULT_BLOCK_SIZE 256
#define DEFAULT_KERNEL_LOOP_ITERATIONS 4096

#define MAX_BLOCK_SIZE 1024

__constant__ int constantA[MAX_BLOCK_SIZE];
__constant__ int constantB[MAX_BLOCK_SIZE];

/* check that all of the entries in a and b match */
__host__ void print_usage(char *name) {
    printf("Comparison Usage: %s <total_num_threads> <threads_per_block> <kernel_loop_iterations>\n", name);
}

/* check that all of the entries in a and b match */
__host__ bool validateResults(int *a, int *b, int N) {
    for (int i = 0; i < N; i++) {
        if (a[i] != b[i]) {
            printf("expected: %d, got: %d, i: %d\n", a[i], b[i], i);
            return false;
        }
    }
    return true;
}

__device__ void doQuadOps(int *inputA, int *inputB, int *output, int threadID, int loop_iterations) {
    for(int i = 0; i < loop_iterations; i++) {
        output[4 * threadID] = inputA[threadID] + inputB[threadID];
        output[(4 * threadID) + 1] = inputA[threadID] - inputB[threadID];
        output[(4 * threadID) + 2] = inputA[threadID] * inputB[threadID];
        output[(4 * threadID) + 3] = inputA[threadID] % inputB[threadID];
    }
    __syncthreads();
}

__device__ void copyToLocal(int *inputA, int *sharedA, int blockOffset, int threadID) {
    sharedA[threadID] = inputA[blockOffset + threadID];
    __syncthreads();
}

__device__ void copyFromLocal(int *sharedOut, int *output, int blockOffset, int threadID) {
    for(int i = 0; i < 4; i++) {
        output[4 * (blockOffset + threadID) + i] = sharedOut[(4 * threadID) + i];
    }
    __syncthreads();
}

// the standard quadKernel using global memory
__global__ void quadKernelGlobal(int *inputA, int *inputB, int *output, int loop_iterations) {
    int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    doQuadOps(inputA, inputB, output, threadID, loop_iterations);
}

// the standard quadKernel using shared memory
__global__ void quadKernelShared(int *inputA, int *inputB, int *output, int loop_iterations) {
    int blockOffset = blockIdx.x * blockDim.x;
    int threadID = threadIdx.x;

    __shared__ int sharedA[MAX_BLOCK_SIZE];
    __shared__ int sharedB[MAX_BLOCK_SIZE];
    __shared__ int sharedOut[4 * MAX_BLOCK_SIZE];

    copyToLocal(inputA, sharedA, blockOffset, threadID);
    copyToLocal(inputB, sharedB, blockOffset, threadID);
    doQuadOps(sharedA, sharedB, sharedOut, threadID, loop_iterations);
    copyFromLocal(sharedOut, output, blockOffset, threadID);
}

// the standard quadKernel using constant memory
__global__ void quadKernelConstant(int *output, int loop_iterations) {
    int blockOffset = blockIdx.x * blockDim.x;
    int threadID = threadIdx.x;

    __shared__ int sharedOut[4 * MAX_BLOCK_SIZE]; // have to output somewhere

    doQuadOps(constantA, constantB, sharedOut, threadID, loop_iterations);
    copyFromLocal(sharedOut, output, blockOffset, threadID);
}

__host__ float doQuadKernelGlobal(int *inputA, int *inputB, int *output, int totalThreads, int blockSize, int loop_iterations) {
    int numBlocks = totalThreads / blockSize;

    // allocate device global memory
    int *device_inputA, *device_inputB, *device_output;
    cudaMalloc(&device_inputA, totalThreads * sizeof(*device_inputA));
    cudaMalloc(&device_inputB, totalThreads * sizeof(*device_inputB));
    cudaMalloc(&device_output, 4 * totalThreads * sizeof(*device_output));

    // setup timing
    float kernelTime;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // copy inputs over
    cudaMemcpy(device_inputA, inputA, totalThreads * sizeof(*device_inputA), cudaMemcpyHostToDevice);
    cudaMemcpy(device_inputB, inputB, totalThreads * sizeof(*device_inputB), cudaMemcpyHostToDevice);

    // execute
    quadKernelGlobal<<<numBlocks, blockSize>>>(device_inputA, device_inputB, device_output, loop_iterations);
    cudaDeviceSynchronize();

    // copy results back
    cudaMemcpy(output, device_output, 4 * totalThreads * sizeof(*output), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&kernelTime, start, stop);

    // free device global memory
    cudaFree(device_inputA);
    cudaFree(device_inputB);
    cudaFree(device_output);

    return kernelTime;
}

__host__ float doQuadKernelShared(int *inputA, int *inputB, int *output, int totalThreads, int blockSize, int loop_iterations) {
    int numBlocks = totalThreads / blockSize;

    // allocate device global memory
    int *device_inputA, *device_inputB, *device_output;
    cudaMalloc(&device_inputA, totalThreads * sizeof(*device_inputA));
    cudaMalloc(&device_inputB, totalThreads * sizeof(*device_inputB));
    cudaMalloc(&device_output, 4 * totalThreads * sizeof(*device_output));

    // setup timing
    float kernelTime;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // copy inputs over
    cudaMemcpy(device_inputA, inputA, totalThreads * sizeof(*device_inputA), cudaMemcpyHostToDevice);
    cudaMemcpy(device_inputB, inputB, totalThreads * sizeof(*device_inputB), cudaMemcpyHostToDevice);

    // execute
    quadKernelShared<<<numBlocks, blockSize>>>(device_inputA, device_inputB, device_output, loop_iterations);
    cudaDeviceSynchronize();

    // copy results back
    cudaMemcpy(output, device_output, 4 * totalThreads * sizeof(*output), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&kernelTime, start, stop);

    // free device global memory
    cudaFree(device_inputA);
    cudaFree(device_inputB);
    cudaFree(device_output);

    return kernelTime;
}

__host__ float doQuadKernelConstant(int *inputA, int *inputB, int *output, int totalThreads, int blockSize, int loop_iterations) {
    int numBlocks = totalThreads / blockSize;

    // allocate device global memory
    int *device_output;
    cudaMalloc(&device_output, 4 * totalThreads * sizeof(*device_output));

    // setup timing
    float kernelTime;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // copy inputs over to constant memory
    cudaMemcpyToSymbol(constantA, inputA, blockSize * sizeof(*inputA));
    cudaMemcpyToSymbol(constantB, inputB, blockSize * sizeof(*inputB));

    // execute
    quadKernelConstant<<<numBlocks, blockSize>>>(device_output, loop_iterations);
    cudaDeviceSynchronize();

    // copy results back
    cudaMemcpy(output, device_output, 4 * totalThreads * sizeof(*output), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&kernelTime, start, stop);

    // free device global memory
    cudaFree(device_output);

    return kernelTime;
}

__host__ void executeComparison(int totalThreads, int blockSize, int loop_iterations) {
    // allocate host arrays
    int *inputA;
    int *inputB;
    int *globalOutput;
    int *sharedOutput;
    int *constantOutput;
    cudaMallocHost(&inputA, totalThreads * sizeof(*inputA));
    cudaMallocHost(&inputB, totalThreads * sizeof(*inputB));
    cudaMallocHost(&globalOutput, 4 * totalThreads * sizeof(*globalOutput)); // times 4 because we do 4 operations...
    cudaMallocHost(&sharedOutput, 4 * totalThreads * sizeof(*sharedOutput));
    cudaMallocHost(&constantOutput, 4 * totalThreads * sizeof(*constantOutput));

    // fill the input arrays with random data
    for (int i = 0; i < totalThreads; i++) {
        inputA[i] = rand();
        inputB[i] = rand();
    }

    // time the kernels using different memory
    float globalTime = doQuadKernelGlobal(inputA, inputB, globalOutput, totalThreads, blockSize, loop_iterations);
    float sharedTime = doQuadKernelShared(inputA, inputB, sharedOutput, totalThreads, blockSize, loop_iterations);
    float constantTime = doQuadKernelConstant(inputA, inputB, constantOutput, totalThreads, blockSize, loop_iterations);

    // verify the outputs
    assert(validateResults(globalOutput, sharedOutput, 4 * totalThreads));
    // constant mem can't store a ton of input, so only check when makes sense
    if(totalThreads == blockSize) {
        assert(validateResults(globalOutput, constantOutput, 4 * totalThreads));
    }

    // print results
    printf("Four operation kernel time... global memory: %f (ms)\n", globalTime);
    printf("Four operation kernel time... shared memory: %f (ms)\n", sharedTime);
    printf("Four operation kernel time... constant memory: %f (ms)\n", constantTime);

    // free memory
    cudaFreeHost(inputA);
    cudaFreeHost(inputB);
    cudaFreeHost(globalOutput);
    cudaFreeHost(sharedOutput);
    cudaFreeHost(constantOutput);
}

__host__ int main(int argc, char **argv) {
    // default arguments
    int totalThreads = DEFAULT_TOTAL_THREADS;
    int blockSize = DEFAULT_BLOCK_SIZE;
    int loop_iterations = DEFAULT_KERNEL_LOOP_ITERATIONS;

    // parse arguments
    if (argc >= 2) {
        totalThreads = atoi(argv[1]);
    }

    if (argc >= 3) {
        blockSize = atoi(argv[2]);
    }

    if (argc >= 4) {
        loop_iterations = atoi(argv[3]);
    }

    if (argc >= 5) {
        print_usage(argv[0]);
        return -1;
    }

    // validate command line arguments
    if(blockSize > MAX_BLOCK_SIZE) {
        blockSize = MAX_BLOCK_SIZE;
        printf("Warning: Block size too large...");
        printf("The block size will be set to %d\n", MAX_BLOCK_SIZE);
    }

    if (totalThreads % blockSize != 0) {
        int numBlocks = (totalThreads + blockSize - 1) / blockSize;
        totalThreads = numBlocks * blockSize;
        printf("Warning: Total thread count is not evenly divisible by the block size...");
        printf("The total number of threads will be rounded up to %d\n", totalThreads);
    }

    // print block and thread count info
    printf("Total Thread Count: %d | Block Size: %d\n", totalThreads, blockSize);
    printf("Kernel operation loop iterations: %d\n", loop_iterations);

    // start the main comparison
    executeComparison(totalThreads, blockSize, loop_iterations);

    return 0;
}