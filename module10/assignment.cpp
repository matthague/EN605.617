#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <CL/cl.h>

const int DEFAULT_ARRAY_SIZE = 1000;

cl_context CreateContext() {
    cl_int errNum;
    cl_uint numPlatforms;
    cl_platform_id firstPlatformId;
    cl_context context = NULL;

    errNum = clGetPlatformIDs(1, &firstPlatformId, &numPlatforms);
    if (errNum != CL_SUCCESS || numPlatforms <= 0) {
        std::cerr << "Failed to find any OpenCL platforms." << std::endl;
        return NULL;
    }

    cl_context_properties contextProperties[] =
            {
                    CL_CONTEXT_PLATFORM,
                    (cl_context_properties) firstPlatformId,
                    0
            };
    context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_GPU,
                                      NULL, NULL, &errNum);
    if (errNum != CL_SUCCESS) {
        std::cout << "Could not create GPU context, trying CPU..." << std::endl;
        context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_CPU,
                                          NULL, NULL, &errNum);
        if (errNum != CL_SUCCESS) {
            std::cerr << "Failed to create an OpenCL GPU or CPU context." << std::endl;
            return NULL;
        }
    }

    return context;
}

cl_command_queue CreateCommandQueue(cl_context context, cl_device_id *device) {
    cl_int errNum;
    cl_device_id *devices;
    cl_command_queue commandQueue = NULL;
    size_t deviceBufferSize = -1;

    // Get the size of the devices buffer
    errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &deviceBufferSize);
    if (errNum != CL_SUCCESS) {
        std::cerr << "Failed call to clGetContextInfo(...,GL_CONTEXT_DEVICES,...)";
        return NULL;
    }

    if (deviceBufferSize <= 0) {
        std::cerr << "No devices available.";
        return NULL;
    }

    // Allocate memory for the devices buffer
    devices = new cl_device_id[deviceBufferSize / sizeof(cl_device_id)];
    errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, deviceBufferSize, devices, NULL);
    if (errNum != CL_SUCCESS) {
        delete[] devices;
        std::cerr << "Failed to get device IDs";
        return NULL;
    }

    commandQueue = clCreateCommandQueue(context, devices[0], 0, NULL);
    if (commandQueue == NULL) {
        delete[] devices;
        std::cerr << "Failed to create commandQueue for device 0";
        return NULL;
    }

    *device = devices[0];
    delete[] devices;
    return commandQueue;
}

//  Create an OpenCL program from the kernel source file
cl_program CreateProgram(cl_context context, cl_device_id device, const char *fileName) {
    cl_int errNum;
    cl_program program;

    std::ifstream kernelFile(fileName, std::ios::in);
    if (!kernelFile.is_open()) {
        std::cerr << "Failed to open file for reading: " << fileName << std::endl;
        return NULL;
    }

    std::ostringstream oss;
    oss << kernelFile.rdbuf();

    std::string srcStdStr = oss.str();
    const char *srcStr = srcStdStr.c_str();
    program = clCreateProgramWithSource(context, 1,
                                        (const char **) &srcStr,
                                        NULL, NULL);
    if (program == NULL) {
        std::cerr << "Failed to create CL program from source." << std::endl;
        return NULL;
    }

    errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (errNum != CL_SUCCESS) {
        // Determine the reason for the error
        char buildLog[16384];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                              sizeof(buildLog), buildLog, NULL);

        std::cerr << "Error in kernel: " << std::endl;
        std::cerr << buildLog;
        clReleaseProgram(program);
        return NULL;
    }

    return program;
}

///
//  Create memory objects used as the arguments to the kernel
//  The kernel takes three arguments: result (output), a (input),
//  and b (input)
//
bool CreateMemObjects(cl_context context, cl_mem memObjects[3],
                      float *a, float *b, int ARRAY_SIZE) {
    memObjects[0] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   sizeof(float) * ARRAY_SIZE, a, NULL);
    memObjects[1] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   sizeof(float) * ARRAY_SIZE, b, NULL);
    memObjects[2] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                   sizeof(float) * ARRAY_SIZE, NULL, NULL);

    if (memObjects[0] == NULL || memObjects[1] == NULL || memObjects[2] == NULL) {
        std::cerr << "Error creating memory objects." << std::endl;
        return false;
    }

    return true;
}

//  Cleanup any created OpenCL resources
void Cleanup(cl_context context, cl_command_queue commandQueue,
             cl_program program, cl_kernel kernel, cl_mem memObjects[3]) {
    for (int i = 0; i < 3; i++) {
        if (memObjects[i] != 0)
            clReleaseMemObject(memObjects[i]);
    }
    if (commandQueue != 0)
        clReleaseCommandQueue(commandQueue);

    if (kernel != 0)
        clReleaseKernel(kernel);

    if (program != 0)
        clReleaseProgram(program);

    if (context != 0)
        clReleaseContext(context);

}

double RunKernel(cl_context *context, cl_program *program, const char *kernel_str, cl_kernel *kernel,
                 cl_command_queue *commandQueue, int ARRAY_SIZE, cl_mem *memObj0, cl_mem *memObj1, cl_mem *memObj2) {
    cl_int errNum;
    cl_mem memObjects[3] = {*memObj0, *memObj1, *memObj2};
    float result[ARRAY_SIZE];

    // Setup timing
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;

    // Create kernel object
    *kernel = clCreateKernel(*program, kernel_str, NULL);
    if (*kernel == NULL) {
        std::cerr << "Failed to create " << kernel_str << std::endl;
        Cleanup(*context, *commandQueue, *program, *kernel, memObjects);
        return -1;
    }

    // Set the kernel arguments (result, a, b)
    errNum = clSetKernelArg(*kernel, 0, sizeof(cl_mem), &memObjects[0]);
    errNum |= clSetKernelArg(*kernel, 1, sizeof(cl_mem), &memObjects[1]);
    errNum |= clSetKernelArg(*kernel, 2, sizeof(cl_mem), &memObjects[2]);
    if (errNum != CL_SUCCESS) {
        std::cerr << "Error setting kernel arguments." << std::endl;
        Cleanup(*context, *commandQueue, *program, *kernel, memObjects);
        return -1;
    }

    size_t globalWorkSize[1] = {ARRAY_SIZE};
    size_t localWorkSize[1] = {1};

    // Start the clock
    auto t1 = high_resolution_clock::now();

    // Queue the kernel up for execution across the array
    errNum = clEnqueueNDRangeKernel(*commandQueue, *kernel, 1, NULL,
                                    globalWorkSize, localWorkSize,
                                    0, NULL, NULL);
    if (errNum != CL_SUCCESS) {
        std::cerr << "Error queuing kernel for execution." << std::endl;
        Cleanup(*context, *commandQueue, *program, *kernel, memObjects);
        return -1;
    }

    // Read the output buffer back to the Host
    errNum = clEnqueueReadBuffer(*commandQueue, memObjects[2], CL_TRUE,
                                 0, ARRAY_SIZE * sizeof(float), result,
                                 0, NULL, NULL);
    if (errNum != CL_SUCCESS) {
        std::cerr << "Error reading result buffer." << std::endl;
        Cleanup(*context, *commandQueue, *program, *kernel, memObjects);
        return -1;
    }

    // Stop the clock
    auto t2 = high_resolution_clock::now();
    duration<double, std::milli> ms_double = t2 - t1;

    // return time in milliseconds
    return ms_double.count();
}

// The main func
int main(int argc, char *argv[]) {
    cl_context context = 0;
    cl_command_queue commandQueue = 0;
    cl_program program = 0;
    cl_device_id device = 0;
    cl_kernel kernel = 0;
    cl_mem memObjects[3] = {0, 0, 0};

    // Get ARRAY_SIZE from args if specified
    int ARRAY_SIZE;
    if (argc > 1) {
        ARRAY_SIZE = atoi(argv[1]);
    } else {
        ARRAY_SIZE = DEFAULT_ARRAY_SIZE;
    }
    std::cout << "Using array size: " << ARRAY_SIZE << std::endl;

    // Create a context on first available platform
    context = CreateContext();
    if (context == NULL) {
        std::cerr << "Failed to create OpenCL context." << std::endl;
        return 1;
    }

    // Create a command-queue on the first device available
    commandQueue = CreateCommandQueue(context, &device);
    if (commandQueue == NULL) {
        Cleanup(context, commandQueue, program, kernel, memObjects);
        return 1;
    }

    // Create program from kernel source
    program = CreateProgram(context, device, "assignment.cl");
    if (program == NULL) {
        Cleanup(context, commandQueue, program, kernel, memObjects);
        return 1;
    }

    // Create memory objects that will be used as arguments to kernel.
    float a[ARRAY_SIZE];
    float b[ARRAY_SIZE];
    for (int i = 0; i < ARRAY_SIZE; i++) {
        a[i] = (float) i;
        b[i] = (float) (i * 2);
    }

    if (!CreateMemObjects(context, memObjects, a, b, ARRAY_SIZE)) {
        Cleanup(context, commandQueue, program, kernel, memObjects);
        return 1;
    }

    // Execute and time each kernel
    double add_time = RunKernel(&context, &program, "add_kernel", &kernel, &commandQueue, ARRAY_SIZE, &memObjects[0],
                                &memObjects[1], &memObjects[2]);
    double sub_time = RunKernel(&context, &program, "sub_kernel", &kernel, &commandQueue, ARRAY_SIZE, &memObjects[0],
                                &memObjects[1], &memObjects[2]);
    double mult_time = RunKernel(&context, &program, "mult_kernel", &kernel, &commandQueue, ARRAY_SIZE, &memObjects[0],
                                 &memObjects[1], &memObjects[2]);
    double div_time = RunKernel(&context, &program, "div_kernel", &kernel, &commandQueue, ARRAY_SIZE, &memObjects[0],
                                &memObjects[1], &memObjects[2]);
    double pow_time = RunKernel(&context, &program, "pow_kernel", &kernel, &commandQueue, ARRAY_SIZE, &memObjects[0],
                                &memObjects[1], &memObjects[2]);

    // Print kernel times
    std::cout << "Addition Kernel: " << add_time << " (ms)..." << std::endl;
    std::cout << "Subtraction Kernel: " << sub_time << " (ms)..." << std::endl;
    std::cout << "Multiplication Kernel: " << mult_time << " (ms)..." << std::endl;
    std::cout << "Division Kernel: " << div_time << " (ms)..." << std::endl;
    std::cout << "Power Kernel: " << pow_time << " (ms)..." << std::endl;

    // Cleanup
    Cleanup(context, commandQueue, program, kernel, memObjects);
    return 0;
}
