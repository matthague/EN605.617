#include <stdio.h>
#include <assert.h>

#define DEFAULT_TOTAL_THREADS 1<<16
#define DEFAULT_BLOCK_SIZE 256
#define DEFAULT_KERNEL_LOOP_ITERATIONS 64

#define MAX_BLOCK_SIZE 1024

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

__global__ void quadKernel(int *inputA, int *inputB, int *output, int loop_iterations) {
    int threadID = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = 0; i < loop_iterations; i++) {
        output[4 * threadID] = inputA[threadID] + inputB[threadID];
        output[(4 * threadID) + 1] = inputA[threadID] - inputB[threadID];
        output[(4 * threadID) + 2] = inputA[threadID] * inputB[threadID];
        output[(4 * threadID) + 3] = inputA[threadID] % inputB[threadID];
    }
}

__host__ float
doQuadKernelRegular(int *inputA, int *inputB, int *output, int totalThreads, int blockSize, int loop_iterations) {
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
    quadKernel<<<numBlocks, blockSize>>>(device_inputA, device_inputB, device_output, loop_iterations);
    cudaDeviceSynchronize();

    // copy results back
    cudaMemcpy(output, device_output, 4 * totalThreads * sizeof(*output), cudaMemcpyDeviceToHost);

    // stop the clock
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&kernelTime, start, stop);

    // free device global memory
    cudaFree(device_inputA);
    cudaFree(device_inputB);
    cudaFree(device_output);

    return kernelTime;
}

__host__ float
doQuadKernelStream(int *inputA, int *inputB, int *output, int totalThreads, int loop_iterations) {
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

    // setup the stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // copy inputs over
    cudaMemcpyAsync(device_inputA, inputA, totalThreads * sizeof(*device_inputA), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(device_inputB, inputB, totalThreads * sizeof(*device_inputB), cudaMemcpyHostToDevice, stream);

    // execute
    quadKernel<<<totalThreads, 1, 1, stream>>>(device_inputA, device_inputB, device_output, loop_iterations);

    // copy results back
    cudaMemcpyAsync(output, device_output, 4 * totalThreads * sizeof(*output), cudaMemcpyDeviceToHost, stream);

    // synchronize the stream
    cudaStreamSynchronize(stream);

    // stop the clock
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&kernelTime, start, stop);

    // free device global memory
    cudaFree(device_inputA);
    cudaFree(device_inputB);
    cudaFree(device_output);

    return kernelTime;
}

__host__ void executeComparison(int totalThreads, int blockSize, int loop_iterations) {
    // allocate host arrays
    int *inputA;
    int *inputB;
    int *regularOutput;
    int *streamOutput;

    cudaHostAlloc(&inputA, totalThreads * sizeof(*inputA), cudaHostAllocDefault);
    cudaHostAlloc(&inputB, totalThreads * sizeof(*inputB), cudaHostAllocDefault);
    cudaHostAlloc(&regularOutput, 4 * totalThreads * sizeof(*regularOutput), cudaHostAllocDefault); // times 4 because we do 4 operations...
    cudaHostAlloc(&streamOutput, 4 * totalThreads * sizeof(*streamOutput), cudaHostAllocDefault);

    // fill the input arrays with random data
    for (int i = 0; i < totalThreads; i++) {
        inputA[i] = rand();
        inputB[i] = rand();
    }

    // time the kernels using different memory
    float regularTime = doQuadKernelRegular(inputA, inputB, regularOutput, totalThreads, blockSize, loop_iterations);
    float streamTime = doQuadKernelStream(inputA, inputB, streamOutput, totalThreads, loop_iterations);

    // verify the outputs
    assert(validateResults(regularOutput, streamOutput, 4 * totalThreads));

    // print results
    printf("Four operation kernel time... using regular kernel: %f (ms)\n", regularTime);
    printf("Four operation kernel time... using streams and events: %f (ms)\n", streamTime);

    // free memory
    cudaFreeHost(inputA);
    cudaFreeHost(inputB);
    cudaFreeHost(regularOutput);
    cudaFreeHost(streamOutput);
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
    if (blockSize > MAX_BLOCK_SIZE) {
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
    printf("Total thread count: %d\n", totalThreads);
    printf("Threads per block for regular kernel: %d\n", blockSize);
    printf("Kernel loop iterations: %d\n\n", loop_iterations);

    // start the main comparison
    executeComparison(totalThreads, blockSize, loop_iterations);

    return 0;
}