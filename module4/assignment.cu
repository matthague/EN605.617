#include <stdio.h>
#include <time.h>
#include <assert.h>

#define DEFAULT_TOTAL_THREADS 1 << 20
#define DEFAULT_BLOCK_SIZE 256

#define MIN_PRINTABLE 32
#define MAX_PRINTABLE 127
#define NUM_ALPHA (MAX_PRINTABLE - MIN_PRINTABLE)

/* check that all of the entries in a and b match */
__host__ void print_usage(char *name) {
    printf("Full Function Usage: %s <total_num_threads> <threads_per_block> <input_file> <key_file>\n", name);
    printf("Comparison Only Usage: %s <total_num_threads> <threads_per_block>\n", name);
}

/* check that all of the entries in a and b match */
__host__ bool validateResults(char *a, char *b, int N) {
    for (int i = 0; i < N; i++) {
        if (a[i] != b[i]) {
            return false;
        }
    }
    return true;
}

__host__ bool validateResults(int *a, int *b, int N) {
    for (int i = 0; i < N; i++) {
        if (a[i] != b[i]) {
            return false;
        }
    }
    return true;
}

__global__ void
fourOperationKernel(int *input, int *output, int additive, int multiplicative, int subtractive, int modulus) {
    int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    output[threadID] = ((multiplicative * (input[threadID] + additive)) - subtractive) % modulus;
}

__host__ double
executeAndTimeFourOpKernel(int *host_input, int *host_output, int totalThreads, int blockSize, int additive,
                           int multiplicative, int subtractive, int modulus) {
    int numBlocks = totalThreads / blockSize;

    // allocate device global memory
    int *device_input, *device_output;
    cudaMalloc(&device_input, totalThreads * sizeof(*device_input));
    cudaMalloc(&device_output, totalThreads * sizeof(*device_output));

    // execute and time
    clock_t startTime = clock();
    cudaMemcpy(device_input, host_input, totalThreads * sizeof(*device_input), cudaMemcpyHostToDevice);
    fourOperationKernel<<<numBlocks, blockSize>>>(device_input, device_output, additive, multiplicative, subtractive,
                                                  modulus);
    cudaDeviceSynchronize();
    cudaMemcpy(host_output, device_output, totalThreads * sizeof(*host_output), cudaMemcpyDeviceToHost);
    clock_t stopTime = clock();
    double kernelTime = (double) (stopTime - startTime) / CLOCKS_PER_SEC;

    // free device global memory
    cudaFree(device_input);
    cudaFree(device_output);

    return kernelTime;
}

__host__ void pageableVsPinnedComparison(int totalThreads, int blockSize) {
    // allocate host arrays
    int *host_pageable_input = (int *) malloc(totalThreads * sizeof(*host_pageable_input));
    int *host_pageable_result = (int *) malloc(totalThreads * sizeof(*host_pageable_result));

    int *host_pinned_input;
    int *host_pinned_result;
    cudaMallocHost(&host_pinned_input, totalThreads * sizeof(*host_pinned_input));
    cudaMallocHost(&host_pinned_result, totalThreads * sizeof(*host_pinned_result));

    // fill the input arrays with data
    for (int i = 0; i < totalThreads; i++) {
        int random_entry = rand();
        host_pageable_input[i] = random_entry;
        host_pinned_input[i] = random_entry;
    }

    // get random kernel parameters for the operations
    int additive = rand();
    int multiplicative = rand();
    int subtractive = rand();
    int modulus = rand();

    // time the kernels using different memory
    double pageable_time = executeAndTimeFourOpKernel(host_pageable_input, host_pageable_result, totalThreads,
                                                      blockSize, additive, multiplicative, subtractive, modulus);
    double pinned_time = executeAndTimeFourOpKernel(host_pinned_input, host_pinned_result, totalThreads, blockSize,
                                                    additive, multiplicative, subtractive, modulus);

    // make sure we computed the same values
    assert(validateResults(host_pageable_result, host_pinned_result, totalThreads));

    // print results
    printf("Four operation kernel times... pageable: %f | pinned: %f  (seconds)\n", pageable_time, pinned_time);

    // free memory
    free(host_pageable_input);
    free(host_pageable_result);
    cudaFreeHost(host_pinned_input);
    cudaFreeHost(host_pinned_result);
}

/* open the file and find its length */
__host__ int getFileLength(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (file == 0) {
        perror("Error: Could not open file...\n");
        return -1;
    }
    fseek(file, 0L, SEEK_END);
    int file_length = ftell(file);
    fclose(file);
    return file_length;
}

/* open the file and read it's contents into memory */
__host__ char *readFile(const char *filename, int length) {
    FILE *file = fopen(filename, "r");
    if (file == 0) {
        perror("Error: Could not open file...\n");
        return NULL;
    }
    char *contents = (char *) malloc(length * sizeof(*contents));
    fgets(contents, length + 1, file);
    fclose(file);
    return contents;
}

__global__ void
caesarCipherKernel(char *device_plaintext, char *device_key, char *device_output, char *device_decrypt_key) {
    int threadID = blockIdx.x * blockDim.x + threadIdx.x;

    char adjusted_plaintext = (device_plaintext[threadID] - MIN_PRINTABLE) % NUM_ALPHA;
    char adjusted_key = (device_key[threadID] - MIN_PRINTABLE) % NUM_ALPHA;
    char adjusted_decrypt_key = (NUM_ALPHA - adjusted_key) % NUM_ALPHA;

    device_output[threadID] = ((adjusted_plaintext + adjusted_key) % NUM_ALPHA) + MIN_PRINTABLE;
    device_decrypt_key[threadID] = adjusted_decrypt_key + MIN_PRINTABLE;
}

__host__ double
executeCaesarKernel(char *host_plaintext, char *host_key, char *host_result, char *host_decrypt_key,
                    int plaintext_length, int blockSize) {
    int numBlocks = (plaintext_length + blockSize - 1) / blockSize;

    // allocate device global memory
    char *device_plaintext, *device_key, *device_output, *device_decrypt_key;
    cudaMalloc(&device_plaintext, plaintext_length * sizeof(*device_plaintext));
    cudaMalloc(&device_key, plaintext_length * sizeof(*device_key));
    cudaMalloc(&device_output, plaintext_length * sizeof(*device_output));
    cudaMalloc(&device_decrypt_key, plaintext_length * sizeof(*device_decrypt_key));

    // execute and time
    clock_t startTime = clock();
    cudaMemcpy(device_plaintext, host_plaintext, plaintext_length * sizeof(*device_plaintext), cudaMemcpyHostToDevice);
    cudaMemcpy(device_key, host_key, plaintext_length * sizeof(*device_key), cudaMemcpyHostToDevice);
    caesarCipherKernel<<<numBlocks, blockSize>>>(device_plaintext, device_key, device_output, device_decrypt_key);
    cudaDeviceSynchronize();
    cudaMemcpy(host_result, device_output, plaintext_length * sizeof(*host_result), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_decrypt_key, device_decrypt_key, plaintext_length * sizeof(*host_decrypt_key),
               cudaMemcpyDeviceToHost);
    clock_t stopTime = clock();
    double kernelTime = (double) (stopTime - startTime) / CLOCKS_PER_SEC;

    // free device global memory
    cudaFree(device_plaintext);
    cudaFree(device_key);
    cudaFree(device_output);
    cudaFree(device_decrypt_key);

    return kernelTime;
}

__host__ void caesarCipher(char *plaintext, char *key, int plaintext_length, int key_length, int blockSize) {
    // allocate host arrays
    char *host_pageable_plaintext = (char *) malloc(plaintext_length * sizeof(*host_pageable_plaintext));
    char *host_pageable_key = (char *) malloc(plaintext_length * sizeof(*host_pageable_key));
    char *host_pageable_result = (char *) malloc(plaintext_length * sizeof(*host_pageable_result));
    char *host_pageable_decrypt_key = (char *) malloc(plaintext_length * sizeof(*host_pageable_decrypt_key));

    char *host_pinned_plaintext;
    char *host_pinned_key;
    char *host_pinned_result;
    char *host_pinned_decrypt_key;
    cudaMallocHost(&host_pinned_plaintext, plaintext_length * sizeof(*host_pinned_plaintext));
    cudaMallocHost(&host_pinned_key, plaintext_length * sizeof(*host_pinned_key));
    cudaMallocHost(&host_pinned_result, plaintext_length * sizeof(*host_pinned_result));
    cudaMallocHost(&host_pinned_decrypt_key, plaintext_length * sizeof(*host_pinned_decrypt_key));

    // fill the input arrays with data
    for (int i = 0; i < plaintext_length; i++) {
        host_pageable_plaintext[i] = plaintext[i];
        host_pinned_plaintext[i] = plaintext[i];

        host_pageable_key[i] = key[i % key_length];
        host_pinned_key[i] = key[i % key_length];
    }

    // call the Caesar cipher kernels using each type of memory
    double pageable_time = executeCaesarKernel(host_pageable_plaintext, host_pageable_key, host_pageable_result,
                                               host_pageable_decrypt_key, plaintext_length, blockSize);
    double pinned_time = executeCaesarKernel(host_pinned_plaintext, host_pinned_key, host_pinned_result,
                                             host_pinned_decrypt_key, plaintext_length, blockSize);

    // make sure we computed the same values for ciphertext and decrypt key
    assert(validateResults(host_pageable_result, host_pinned_result, plaintext_length));
    assert(validateResults(host_pageable_decrypt_key, host_pinned_decrypt_key, plaintext_length));

    // do a test decryption to recover the original plaintext
    executeCaesarKernel(host_pageable_result, host_pageable_decrypt_key, host_pageable_plaintext, host_pageable_key,
                        plaintext_length, blockSize);

    // print results
    printf("Caesar cipher operation kernel times... pageable: %f | pinned: %f  (seconds)\n", pageable_time,
           pinned_time);
    printf("\nGiven Plaintext: ");
    for (int i = 0; i < plaintext_length; i++) {
        printf("%c", host_pinned_plaintext[i]);
    }
    printf("\n\nGiven Key: ");
    for (int i = 0; i < key_length; i++) {
        printf("%c", host_pinned_key[i]);
    }
    printf("\n\nCiphertext: ");
    for (int i = 0; i < plaintext_length; i++) {
        printf("%c", host_pinned_result[i]);
    }
    printf("\n\nDecryption Key: ");
    for (int i = 0; i < key_length; i++) {
        printf("%c", host_pinned_decrypt_key[i]);
    }
    printf("\n\nRecovered Plaintext: ");
    for (int i = 0; i < plaintext_length; i++) {
        printf("%c", host_pageable_plaintext[i]);
    }
    printf("\n");

    // free memory
    free(host_pageable_plaintext);
    free(host_pageable_key);
    free(host_pageable_result);
    free(host_pageable_decrypt_key);
    cudaFreeHost(host_pinned_plaintext);
    cudaFreeHost(host_pinned_key);
    cudaFreeHost(host_pinned_result);
    cudaFreeHost(host_pinned_decrypt_key);
}

int main(int argc, char **argv) {
    // default arguments
    int totalThreads = DEFAULT_TOTAL_THREADS;
    int blockSize = DEFAULT_BLOCK_SIZE;

    // parse arguments
    if (argc >= 2) {
        totalThreads = atoi(argv[1]);
    }

    if (argc >= 3) {
        blockSize = atoi(argv[2]);
    }

    if (argc == 4 || argc > 5) {
        print_usage(argv[0]);
        return -1;
    }

    // validate command line arguments
    if (totalThreads % blockSize != 0) {
        int numBlocks = (totalThreads + blockSize - 1) / blockSize;
        totalThreads = numBlocks * blockSize;

        printf("Warning: Total thread count is not evenly divisible by the block size\n");
        printf("The total number of threads will be rounded up to %d\n", totalThreads);
    }

    // print block and thread count info
    printf("Total Thread Count: %d | Block Size: %d\n", totalThreads, blockSize);

    // compare paged and pinned memory performance
    pageableVsPinnedComparison(totalThreads, blockSize);

    // do caesar cipher operations if the args are provided
    if (argc == 5) {
        // load plaintext
        int plaintext_length = getFileLength(argv[3]);
        if (plaintext_length <= 0) {
            printf("Error: Plaintext size is invalid...\n");
            return -1;
        }
        char *plaintext = readFile(argv[3], plaintext_length);

        // load key
        int key_length = getFileLength(argv[4]);
        if(key_length > plaintext_length) {
            // the effective length of the key
            key_length = plaintext_length;
        }
        if (key_length <= 0) {
            printf("Error: Key size is invalid...\n");
            return -1;
        }
        char *key = readFile(argv[4], key_length);

        // do cipher operations
        caesarCipher(plaintext, key, plaintext_length, key_length, blockSize);

        free(plaintext);
        free(key);
    }

    return 0;
}