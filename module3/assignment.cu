#include <stdio.h>
#include <time.h>
#include <assert.h>

__global__ void cudaAdd(int *a, int *b, int *c) {
    int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    c[threadID] = a[threadID] + b[threadID];
}

__global__ void cudaSub(int *a, int *b, int *c) {
    int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    c[threadID] = a[threadID] - b[threadID];
}

__global__ void cudaMult(int *a, int *b, int *c) {
    int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    c[threadID] = a[threadID] * b[threadID];
}

__global__ void cudaMod(int *a, int *b, int *c) {
    int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    c[threadID] = a[threadID] % b[threadID];
}

__global__ void cudaAlgorithmWithoutDivergence(int *a, int *b, int *c) {
    int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    int offset =  2*(threadID % 2) - 1; // a way to avoid divergence
    c[threadID] = a[threadID] ^ b[threadID] + offset;
}

__global__ void cudaAlgorithmWithDivergence(int *a, int *b, int *c) {
    int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    c[threadID] = a[threadID] ^ b[threadID];

    // the source of the divergence
    if (threadID % 32 == 0) {
        c[threadID] -= 1;
    } else if (threadID % 32 == 1) {
        c[threadID] += 1;
    } else if (threadID % 32 == 2) {
        c[threadID] -= 1;
    } else if (threadID % 32 == 3) {
        c[threadID] += 1;
    } else if (threadID % 32 == 4) {
        c[threadID] -= 1;
    } else if (threadID % 32 == 5) {
        c[threadID] += 1;
    } else if (threadID % 32 == 6) {
        c[threadID] -= 1;
    } else if (threadID % 32 == 7) {
        c[threadID] += 1;
    } else if (threadID % 32 == 8) {
        c[threadID] -= 1;
    } else if (threadID % 32 == 9) {
        c[threadID] += 1;
    } else if (threadID % 32 == 10) {
        c[threadID] -= 1;
    } else if (threadID % 32 == 11) {
        c[threadID] += 1;
    } else if (threadID % 32 == 12) {
        c[threadID] -= 1;
    } else if (threadID % 32 == 13) {
        c[threadID] += 1;
    } else if (threadID % 32 == 14) {
        c[threadID] -= 1;
    } else if (threadID % 32 == 15) {
        c[threadID] += 1;
    } else if (threadID % 32 == 16) {
        c[threadID] -= 1;
    } else if (threadID % 32 == 17) {
        c[threadID] += 1;
    } else if (threadID % 32 == 18) {
        c[threadID] -= 1;
    } else if (threadID % 32 == 19) {
        c[threadID] += 1;
    } else if (threadID % 32 == 20) {
        c[threadID] -= 1;
    } else if (threadID % 32 == 21) {
        c[threadID] += 1;
    } else if (threadID % 32 == 22) {
        c[threadID] -= 1;
    } else if (threadID % 32 == 23) {
        c[threadID] += 1;
    } else if (threadID % 32 == 24) {
        c[threadID] -= 1;
    } else if (threadID % 32 == 25) {
        c[threadID] += 1;
    } else if (threadID % 32 == 26) {
        c[threadID] -= 1;
    } else if (threadID % 32 == 27) {
        c[threadID] += 1;
    } else if (threadID % 32 == 28) {
        c[threadID] -= 1;
    } else if (threadID % 32 == 29) {
        c[threadID] += 1;
    } else if (threadID % 32 == 30) {
        c[threadID] -= 1;
    } else {
        c[threadID] += 1;
    }
}

void hostAdd(int *a, int *b, int *c, int N) {
    for (int i = 0; i < N; i++) {
        c[i] = a[i] + b[i];
    }
}

void hostSub(int *a, int *b, int *c, int N) {
    for (int i = 0; i < N; i++) {
        c[i] = a[i] - b[i];
    }
}

void hostMult(int *a, int *b, int *c, int N) {
    for (int i = 0; i < N; i++) {
        c[i] = a[i] * b[i];
    }
}

void hostMod(int *a, int *b, int *c, int N) {
    for (int i = 0; i < N; i++) {
        c[i] = a[i] % b[i];
    }
}

void hostAlgorithmWithoutDivergence(int *a, int *b, int *c, int N) {
    for (int i = 0; i < N; i++) {
        int offset =  2*(i % 2) - 1;
        c[i] = a[i] ^ b[i] + offset;
    }
}

void hostAlgorithmWithDivergence(int *a, int *b, int *c, int N) {
    for (int i = 0; i < N; i++) {
        c[i] = a[i] ^ b[i];
        if(i % 2 == 0) {
            c[i] -= 1;
        } else {
            c[i] += 1;
        }
    }
}

/* check that all of the entries in a and b match */
bool validateResults(int *a, int *b, int N) {
    for (int i = 0; i < N; i++) {
        if (a[i] != b[i]) {
            return false;
        }
    }
    return true;
}

int main(int argc, char **argv) {
    /*                 */
    /* PARSE ARGUMENTS */
    /*                 */

    // default arguments
    int totalThreads = (1 << 16);
    int blockSize = 256;

    // read command line arguments
    if (argc >= 2) {
        totalThreads = atoi(argv[1]);
    }

    if (argc >= 3) {
        blockSize = atoi(argv[2]);
    }

    int numBlocks = totalThreads / blockSize;

    // validate command line arguments
    if (totalThreads % blockSize != 0) {
        numBlocks++;
        totalThreads = numBlocks * blockSize;

        printf("Warning: Total thread count is not evenly divisible by the block size\n");
        printf("The total number of threads will be rounded up to %d\n", totalThreads);
    }

    /*              */
    /* SETUP ARRAYS */
    /*              */

    // allocate host arrays
    int *host_a = (int *) malloc(totalThreads * sizeof(*host_a));
    int *host_b = (int *) malloc(totalThreads * sizeof(*host_b));

    int *device_add_results = (int *) malloc(totalThreads * sizeof(*device_add_results));
    int *host_add_results = (int *) malloc(totalThreads * sizeof(*host_add_results));

    int *device_sub_results = (int *) malloc(totalThreads * sizeof(*device_sub_results));
    int *host_sub_results = (int *) malloc(totalThreads * sizeof(*host_sub_results));

    int *device_mult_results = (int *) malloc(totalThreads * sizeof(*device_mult_results));
    int *host_mult_results = (int *) malloc(totalThreads * sizeof(*host_mult_results));

    int *device_mod_results = (int *) malloc(totalThreads * sizeof(*device_mod_results));
    int *host_mod_results = (int *) malloc(totalThreads * sizeof(*host_mod_results));

    int *device_withoutdivergence_results = (int *) malloc(totalThreads * sizeof(*device_withoutdivergence_results));
    int *host_withoutdivergence_results = (int *) malloc(totalThreads * sizeof(*host_withoutdivergence_results));

    int *device_withdivergence_results = (int *) malloc(totalThreads * sizeof(*device_withdivergence_results));
    int *host_withdivergence_results = (int *) malloc(totalThreads * sizeof(*host_withdivergence_results));

    // allocate device arrays
    int *dev_a, *dev_b, *dev_c;
    cudaMalloc(&dev_a, totalThreads * sizeof(*dev_a));
    cudaMalloc(&dev_b, totalThreads * sizeof(*dev_b));
    cudaMalloc(&dev_c, totalThreads * sizeof(*dev_c));

    // fill the input arrays with data
    for (int i = 0; i < totalThreads; i++) {
        host_a[i] = i; // 0 - totalThreads
        host_b[i] = (rand() % 2) + 1; // random values between 1-3
    }

    // copy inputs to the device
    cudaMemcpy(dev_a, host_a, totalThreads * sizeof(*dev_a), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, host_b, totalThreads * sizeof(*dev_b), cudaMemcpyHostToDevice);

    /*             */
    /*  ADDITION   */
    /*             */

    // execute and time addition on the host
    clock_t hostAddStart = clock();
    hostAdd(host_a, host_b, host_add_results, totalThreads);
    clock_t hostAddStop = clock();
    double hostAddElapsedTime = (double) (hostAddStop - hostAddStart) / CLOCKS_PER_SEC;

    // execute and time the device add kernel
    clock_t cudaAddStart = clock();
    cudaAdd<<<numBlocks, blockSize>>>(dev_a, dev_b, dev_c);
    cudaDeviceSynchronize();
    clock_t cudaAddStop = clock();
    double cudaAddElapsedTime = (double) (cudaAddStop - cudaAddStart) / CLOCKS_PER_SEC;

    // copy device addition results back to the host and verify them
    cudaMemcpy(device_add_results, dev_c, totalThreads * sizeof(*device_add_results), cudaMemcpyDeviceToHost);
    assert(validateResults(device_add_results, host_add_results, totalThreads) == true);

    /*             */
    /* SUBTRACTION */
    /*             */

    // execute and time subtraction on the host
    clock_t hostSubStart = clock();
    hostSub(host_a, host_b, host_sub_results, totalThreads);
    clock_t hostSubStop = clock();
    double hostSubElapsedTime = (double) (hostSubStop - hostSubStart) / CLOCKS_PER_SEC;

    // execute and time the device add kernel
    clock_t cudaSubStart = clock();
    cudaSub<<<numBlocks, blockSize>>>(dev_a, dev_b, dev_c);
    cudaDeviceSynchronize();
    clock_t cudaSubStop = clock();
    double cudaSubElapsedTime = (double) (cudaSubStop - cudaSubStart) / CLOCKS_PER_SEC;

    // copy device addition results back to the host and verify them
    cudaMemcpy(device_sub_results, dev_c, totalThreads * sizeof(*device_sub_results), cudaMemcpyDeviceToHost);
    assert(validateResults(device_sub_results, host_sub_results, totalThreads) == true);

    /*             */
    /*  MULTIPLY   */
    /*             */

    // execute and time subtraction on the host
    clock_t hostMultStart = clock();
    hostMult(host_a, host_b, host_mult_results, totalThreads);
    clock_t hostMultStop = clock();
    double hostMultElapsedTime = (double) (hostMultStop - hostMultStart) / CLOCKS_PER_SEC;

    // execute and time the device add kernel
    clock_t cudaMultStart = clock();
    cudaMult<<<numBlocks, blockSize>>>(dev_a, dev_b, dev_c);
    cudaDeviceSynchronize();
    clock_t cudaMultStop = clock();
    double cudaMultElapsedTime = (double) (cudaMultStop - cudaMultStart) / CLOCKS_PER_SEC;

    // copy device addition results back to the host and verify them
    cudaMemcpy(device_mult_results, dev_c, totalThreads * sizeof(*device_mult_results), cudaMemcpyDeviceToHost);
    assert(validateResults(device_mult_results, host_mult_results, totalThreads) == true);

    /*             */
    /*   MODULUS   */
    /*             */

    // execute and time subtraction on the host
    clock_t hostModStart = clock();
    hostMod(host_a, host_b, host_mod_results, totalThreads);
    clock_t hostModStop = clock();
    double hostModElapsedTime = (double) (hostModStop - hostModStart) / CLOCKS_PER_SEC;

    // execute and time the device add kernel
    clock_t cudaModStart = clock();
    cudaMod<<<numBlocks, blockSize>>>(dev_a, dev_b, dev_c);
    cudaDeviceSynchronize();
    clock_t cudaModStop = clock();
    double cudaModElapsedTime = (double) (cudaModStop - cudaModStart) / CLOCKS_PER_SEC;

    // copy device addition results back to the host and verify them
    cudaMemcpy(device_mod_results, dev_c, totalThreads * sizeof(*device_mod_results), cudaMemcpyDeviceToHost);
    assert(validateResults(device_mod_results, host_mod_results, totalThreads) == true);

    /*                              */
    /*   ALGORITHM W/O DIVERGENCE   */
    /*                              */

    // execute and time subtraction on the host
    clock_t hostWithoutDivergenceStart = clock();
    hostAlgorithmWithoutDivergence(host_a, host_b, host_withoutdivergence_results, totalThreads);
    clock_t hostWithoutDivergenceStop = clock();
    double hostWithoutDivergenceElapsedTime = (double) (hostWithoutDivergenceStop - hostWithoutDivergenceStart) / CLOCKS_PER_SEC;

    // execute and time the device add kernel
    clock_t cudaWithoutDivergenceStart = clock();
    cudaAlgorithmWithoutDivergence<<<numBlocks, blockSize>>>(dev_a, dev_b, dev_c);
    cudaDeviceSynchronize();
    clock_t cudaWithoutDivergenceStop = clock();
    double cudaWithoutDivergenceElapsedTime = (double) (cudaWithoutDivergenceStop - cudaWithoutDivergenceStart) / CLOCKS_PER_SEC;

    // copy device addition results back to the host and verify them
    cudaMemcpy(device_withoutdivergence_results, dev_c, totalThreads * sizeof(*device_withoutdivergence_results), cudaMemcpyDeviceToHost);
    assert(validateResults(device_withoutdivergence_results, host_withoutdivergence_results, totalThreads) == true);

    /*                               */
    /*   ALGORITHM WITH DIVERGENCE   */
    /*                               */

    // execute and time subtraction on the host
    clock_t hostWithDivergenceStart = clock();
    hostAlgorithmWithDivergence(host_a, host_b, host_withdivergence_results, totalThreads);
    clock_t hostWithDivergenceStop = clock();
    double hostWithDivergenceElapsedTime = (double) (hostWithDivergenceStop - hostWithDivergenceStart) / CLOCKS_PER_SEC;

    // execute and time the device add kernel
    clock_t cudaWithDivergenceStart = clock();
    cudaAlgorithmWithDivergence<<<numBlocks, blockSize>>>(dev_a, dev_b, dev_c);
    cudaDeviceSynchronize();
    clock_t cudaWithDivergenceStop = clock();
    double cudaWithDivergenceElapsedTime = (double) (cudaWithDivergenceStop - cudaWithDivergenceStart) / CLOCKS_PER_SEC;

    // copy device addition results back to the host and verify them
    cudaMemcpy(device_withdivergence_results, dev_c, totalThreads * sizeof(*device_withdivergence_results), cudaMemcpyDeviceToHost);
    assert(validateResults(device_withdivergence_results, host_withdivergence_results, totalThreads) == true);

    /*                */
    /* OUTPUT RESULTS */
    /*                */

    // display block/thread info
    printf("Total number of threads: %d\n", totalThreads);
    printf("Threads per block: %d\n", blockSize);
    printf("Number of blocks: %d\n\n", numBlocks);

    // display times for standard operations
    printf("Addition times... Host: %f | Device: %f  (seconds)\n", hostAddElapsedTime, cudaAddElapsedTime);
    printf("Subtraction times... Host: %f | Device: %f  (seconds)\n", hostSubElapsedTime, cudaSubElapsedTime);
    printf("Multiplication times... Host: %f | Device: %f  (seconds)\n", hostMultElapsedTime, cudaMultElapsedTime);
    printf("Modulation times... Host: %f | Device: %f  (seconds)\n\n", hostModElapsedTime, cudaModElapsedTime);

    // display times for divergence tests
    printf("Demo algorithm without divergence times... Host: %f | Device: %f  (seconds)\n", hostWithoutDivergenceElapsedTime, cudaWithoutDivergenceElapsedTime);
    printf("Demo algorithm with divergence times... Host: %f | Device: %f  (seconds)\n", hostWithDivergenceElapsedTime, cudaWithDivergenceElapsedTime);

    /*                */
    /*  CLEANUP/FREE  */
    /*                */

    // free device memory
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    // free host memory
    free(host_a);
    free(host_b);

    free(device_add_results);
    free(host_add_results);

    free(device_sub_results);
    free(host_sub_results);

    free(device_mult_results);
    free(host_mult_results);

    free(device_mod_results);
    free(host_mod_results);

    free(device_withoutdivergence_results);
    free(host_withoutdivergence_results);

    free(device_withdivergence_results);
    free(host_withdivergence_results);

    return 0;
}
