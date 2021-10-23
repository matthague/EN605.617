#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#define DEFAULT_MATRIX_SIZE 256
#define DEFAULT_BLOCK_SIZE 256

void printUsage(char* argv[]) {
    printf("Usage: %s <matrix_size> <:OPTIONAL: threads_per_block>", argv[0]);
}

__global__ void initializePRNGKernel(curandState *state, int seed, int maxIndex){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < maxIndex) {
        curand_init((seed << 20) + idx, 0, 0, &state[idx]);
    }
}

__global__ void getRandomMatrixKernel(float* resultMatrix, curandState *state, int maxIndex) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < maxIndex) {
        resultMatrix[idx] = 255.0 * curand_uniform(&state[idx]);
    }
}

float makeRandomMatrix(float* inputMatrix, int matrixSize, int blockSize) {
    int totalElements = matrixSize * matrixSize;
    int seed = time(0);
    int numBlocks = (totalElements + blockSize - 1) / blockSize;

    // allocate resources
    curandState *rngState;
    cudaMalloc(&rngState, totalElements * sizeof(*rngState));
    float *deviceMatrix;
    cudaMalloc(&deviceMatrix, totalElements * sizeof(*deviceMatrix));

    // setup timing
    float kernelTime;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // setup the rng
    initializePRNGKernel<<<numBlocks, blockSize>>>(rngState, seed, totalElements);
    cudaDeviceSynchronize();

    // generate random matrix and copy it back
    getRandomMatrixKernel<<<numBlocks, blockSize>>>(deviceMatrix, rngState, totalElements);
    cudaDeviceSynchronize();
    cudaMemcpy(inputMatrix, deviceMatrix, totalElements * sizeof(*inputMatrix), cudaMemcpyDeviceToHost);

    // stop the clock
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&kernelTime, start, stop);

    // free resources
    cudaFree(rngState);
    cudaFree(deviceMatrix);

    // return time
    return kernelTime;
}

float getSingularValues(float* inputMatrix, int matrixSize, int blockSize) {
    // setup solver handles
    cusolverDnHandle_t cusolverH = NULL;
    cublasHandle_t cublasH = NULL;

    cusolverDnCreate(&cusolverH);
    cublasCreate(&cublasH);

    // constant dimensions for column major matrix
    const int m = matrixSize;
    const int n = matrixSize;
    const int lda = m;

    // result array
    float *S = NULL; // [n] singular values
    cudaMallocHost(&S, n * (sizeof(*S)));

    // setup device arrays
    float *d_A = NULL;
    float *d_S = NULL;
    float *d_U = NULL;
    float *d_VT = NULL;
    int *devInfo = NULL;
    float *d_work = NULL;
    float *d_rwork = NULL;

    cudaMalloc(&d_A, sizeof(float) * lda * n);
    cudaMalloc(&d_S, sizeof(float) * n);
    cudaMalloc(&d_U, sizeof(float) * lda * m);
    cudaMalloc(&d_VT, sizeof(float) * lda * n);
    cudaMalloc(&devInfo, sizeof(int));

    // setup timing
    float kernelTime;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // copy input array to device
    cudaMemcpy(d_A, inputMatrix, sizeof(float) * lda * n, cudaMemcpyHostToDevice);

    // query working space of SVD solver
    int lwork = 0;
    cusolverDnSgesvd_bufferSize(cusolverH, m, n, &lwork);
    cudaMalloc(&d_work, sizeof(float) * lwork);

    // compute SVD
    signed char jobu = 'A'; // all m columns of U
    signed char jobvt = 'A'; // all n columns of VT
    cusolverDnSgesvd(
            cusolverH,
            jobu,
            jobvt,
            m,
            n,
            d_A,
            lda,
            d_S,
            d_U,
            lda,  // ldu
            d_VT,
            lda, // ldvt,
            d_work,
            lwork,
            d_rwork,
            devInfo);

    cudaDeviceSynchronize();

    // copy singular value results back
    cudaMemcpy(S, d_S, sizeof(float) * n, cudaMemcpyDeviceToHost);

    // stop the clock
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&kernelTime, start, stop);

    // free resources
    cublasDestroy(cublasH);
    cusolverDnDestroy(cusolverH);

    cudaFree(devInfo);
    cudaFree(d_rwork);
    cudaFree(d_work);

    cudaFree(d_A);
    cudaFree(d_S);
    cudaFree(d_U);
    cudaFree(d_VT);

    cudaFreeHost(S);

    return kernelTime;
}

int main(int argc, char* argv[]) {
    // parse args
    if(argc > 3) {
        printUsage(argv);
	return -1;
    }

    int matrixSize = DEFAULT_MATRIX_SIZE;
    if(argc > 1) {
        matrixSize = atoi(argv[1]);
    }

    int blockSize = DEFAULT_BLOCK_SIZE;
    if(argc > 2) {
        blockSize = atoi(argv[2]);
    }

    // allocate space for the matrix
    float* randomMatrix;
    cudaMallocHost(&randomMatrix, matrixSize * matrixSize * sizeof(*randomMatrix));

    // use curand to generate random matrix
    float randomTime = makeRandomMatrix(randomMatrix, matrixSize, blockSize);

    // use cusolver to find singular values of the random matrix
    float svdTime = getSingularValues(randomMatrix, matrixSize, blockSize);

    // print results
    printf("Matrix Dimension: %d -- Threads Per Block: %d\n", matrixSize, blockSize);
    printf("PRNG operation time... : %f (ms)\n", randomTime);
    printf("Singular value operation time... : %f (ms)\n", svdTime);

    // free remaining resources
    cudaFree(randomMatrix);

    return 0;
}
