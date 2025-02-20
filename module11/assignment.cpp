//
// Book:      OpenCL(R) Programming Guide
// Authors:   Aaftab Munshi, Benedict Gaster, Timothy Mattson, James Fung, Dan Ginsburg
// ISBN-10:   0-321-74964-2
// ISBN-13:   978-0-321-74964-2
// Publisher: Addison-Wesley Professional
// URLs:      http://safari.informit.com/9780132488006/
//            http://www.openclprogrammingguide.com
//


// Convolution.cpp
//
//    This is a simple example that demonstrates OpenCL platform, device, and context
//    use.

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else

#include <CL/cl.h>

#endif

#if !defined(CL_CALLBACK)
#define CL_CALLBACK
#endif

// Constants
const unsigned int inputSignalWidth = 49;
const unsigned int inputSignalHeight = 49;
cl_uint inputSignal[inputSignalHeight][inputSignalWidth];

const unsigned int maskWidth = 7;
const unsigned int maskHeight = 7;
cl_float mask[maskHeight][maskWidth] =
        {
                {.25,    .25,    .33333, .33333, .33333, .25,    .25},
                {.25,    .33333, .5,     .5,     .5,     .33333, .25},
                {.33333, .5,     1,      1,      1,      .5,     .33333},
                {.33333, .5,     1,      0,      1,      .5,     .33333},
                {.33333, .5,     1,      1,      1,      .5,     .33333},
                {.25,    .33333, .5,     .5,     .5,     .33333, .25},
                {.25,    .25,    .33333, .33333, .33333, .25,    .25}
        };

const unsigned int outputSignalWidth = inputSignalWidth - (maskWidth - 1);
const unsigned int outputSignalHeight = inputSignalHeight - (maskHeight - 1);
cl_uint outputSignal[outputSignalHeight][outputSignalWidth];

// Function to check and handle OpenCL errors
inline void
checkErr(cl_int err, const char *name) {
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: " << name << " (" << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

void CL_CALLBACK contextCallback(
        const char *errInfo,
        const void *private_info,
        size_t cb,
        void *user_data) {
    std::cout << "Error occured during context use: " << errInfo << std::endl;
    exit(1);
}

int main(int argc, char **argv) {
    cl_int errNum;
    cl_uint numPlatforms;
    cl_uint numDevices;
    cl_platform_id *platformIDs;
    cl_device_id *deviceIDs;
    cl_context context = NULL;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_mem inputSignalBuffer;
    cl_mem outputSignalBuffer;
    cl_mem maskBuffer;

    // Set input signal as a random integer between 0 and 99
    srand(time(0));
    for (unsigned int i = 0; i < inputSignalHeight; i++) {
        for (unsigned int j = 0; j < inputSignalWidth; j++) {
            inputSignal[i][j] = rand() % 100;
        }
    }

    // First, select an OpenCL platform to run on.
    errNum = clGetPlatformIDs(0, NULL, &numPlatforms);
    checkErr(
            (errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS),
            "clGetPlatformIDs");

    platformIDs = (cl_platform_id *) alloca(
            sizeof(cl_platform_id) * numPlatforms);

    errNum = clGetPlatformIDs(numPlatforms, platformIDs, NULL);
    checkErr(
            (errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS),
            "clGetPlatformIDs");

    // Find a GPU device
    deviceIDs = NULL;
    cl_uint i;
    for (i = 0; i < numPlatforms; i++) {
        errNum = clGetDeviceIDs(
                platformIDs[i],
                CL_DEVICE_TYPE_GPU,
                0,
                NULL,
                &numDevices);
        if (errNum != CL_SUCCESS && errNum != CL_DEVICE_NOT_FOUND) {
            checkErr(errNum, "clGetDeviceIDs");
        } else if (numDevices > 0) {
            deviceIDs = (cl_device_id *) alloca(sizeof(cl_device_id) * numDevices);
            errNum = clGetDeviceIDs(
                    platformIDs[i],
                    CL_DEVICE_TYPE_GPU,
                    numDevices,
                    &deviceIDs[0],
                    NULL);
            checkErr(errNum, "clGetDeviceIDs");
            break;
        }
    }

    // Create an OpenCL context
    cl_context_properties contextProperties[] =
            {
                    CL_CONTEXT_PLATFORM,
                    (cl_context_properties) platformIDs[i],
                    0
            };
    context = clCreateContext(
            contextProperties,
            numDevices,
            deviceIDs,
            &contextCallback,
            NULL,
            &errNum);
    checkErr(errNum, "clCreateContext");

    std::ifstream srcFile("assignment.cl");
    checkErr(srcFile.is_open() ? CL_SUCCESS : -1, "reading assignment.cl");

    std::string srcProg(
            std::istreambuf_iterator<char>(srcFile),
            (std::istreambuf_iterator<char>()));

    const char *src = srcProg.c_str();
    size_t length = srcProg.length();

    // Create program from source
    program = clCreateProgramWithSource(
            context,
            1,
            &src,
            &length,
            &errNum);
    checkErr(errNum, "clCreateProgramWithSource");

    // Build program
    errNum = clBuildProgram(
            program,
            numDevices,
            deviceIDs,
            NULL,
            NULL,
            NULL);
    if (errNum != CL_SUCCESS) {
        // Determine the reason for the error
        char buildLog[16384];
        clGetProgramBuildInfo(
                program,
                deviceIDs[0],
                CL_PROGRAM_BUILD_LOG,
                sizeof(buildLog),
                buildLog,
                NULL);

        std::cerr << "Error in kernel: " << std::endl;
        std::cerr << buildLog;
        checkErr(errNum, "clBuildProgram");
    }

    // Create kernel object
    kernel = clCreateKernel(
            program,
            "convolve",
            &errNum);
    checkErr(errNum, "clCreateKernel");

    // Allocate device buffers
    inputSignalBuffer = clCreateBuffer(
            context,
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            sizeof(cl_uint) * inputSignalHeight * inputSignalWidth,
            static_cast<void *>(inputSignal),
            &errNum);
    checkErr(errNum, "clCreateBuffer(inputSignal)");

    maskBuffer = clCreateBuffer(
            context,
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            sizeof(cl_float) * maskHeight * maskWidth,
            static_cast<void *>(mask),
            &errNum);
    checkErr(errNum, "clCreateBuffer(mask)");

    outputSignalBuffer = clCreateBuffer(
            context,
            CL_MEM_WRITE_ONLY,
            sizeof(cl_uint) * outputSignalHeight * outputSignalWidth,
            NULL,
            &errNum);
    checkErr(errNum, "clCreateBuffer(outputSignal)");

    // Pick the first device and create command queue.
    queue = clCreateCommandQueue(
            context,
            deviceIDs[0],
            0,
            &errNum);
    checkErr(errNum, "clCreateCommandQueue");

    errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputSignalBuffer);
    errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &maskBuffer);
    errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &outputSignalBuffer);
    errNum |= clSetKernelArg(kernel, 3, sizeof(cl_uint), &inputSignalWidth);
    errNum |= clSetKernelArg(kernel, 4, sizeof(cl_uint), &maskWidth);
    checkErr(errNum, "clSetKernelArg");

    const size_t globalWorkSize[2] = {outputSignalWidth, outputSignalHeight};
    const size_t localWorkSize[2] = {1, 1};

    // Start the clock
    auto t1 = std::chrono::high_resolution_clock::now();

    // Queue the kernel up for execution across the array
    errNum = clEnqueueNDRangeKernel(
            queue,
            kernel,
            2,
            NULL,
            globalWorkSize,
            localWorkSize,
            0,
            NULL,
            NULL);
    checkErr(errNum, "clEnqueueNDRangeKernel");

    // Stop the clock
    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1);

    // Read the results
    errNum = clEnqueueReadBuffer(
            queue,
            outputSignalBuffer,
            CL_TRUE,
            0,
            sizeof(cl_uint) * outputSignalHeight * outputSignalHeight,
            outputSignal,
            0,
            NULL,
            NULL);
    checkErr(errNum, "clEnqueueReadBuffer");

    // Output the result buffer
    for (int y = 0; y < outputSignalHeight; y++) {
        for (int x = 0; x < outputSignalWidth; x++) {
            std::cout << outputSignal[y][x] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl << "Executed program succesfully." << std::endl;

    // Output timing info
    std::cout << "Kernel took: " << duration.count() << " (microseconds)" << std::endl;

    return 0;
}
