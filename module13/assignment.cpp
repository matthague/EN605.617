#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <stdlib.h>
#include <time.h>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else

#include <CL/cl.h>

#endif

#define DEFAULT_LIST_SIZE 16
#define MAX_ELEMENT_SIZE 1000

void printList(int *list, int list_size) {
    for (int i = 0; i < list_size; i++) {
        std::cout << list[i] << " ";
    }
    std::cout << std::endl;
}

bool checkSorted(int *list, int list_size) {
    for (int i = 0; i < list_size - 1; i++) {
        if (list[i] > list[i + 1]) {
            return false;
        }
    }
    return true;
}

cl_context CreateContext() {
    cl_int errNum;
    cl_uint numPlatforms;
    cl_platform_id firstPlatformId;
    cl_context context = NULL;

    // First, select an OpenCL platform to run on
    errNum = clGetPlatformIDs(1, &firstPlatformId, &numPlatforms);
    if (errNum != CL_SUCCESS || numPlatforms <= 0) {
        std::cerr << "Failed to find any OpenCL platforms." << std::endl;
        return NULL;
    }

    // Create the actual context
    cl_context_properties contextProperties[] = {
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

    // First get the size of the devices buffer
    errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &deviceBufferSize);
    if (errNum != CL_SUCCESS) {
        std::cerr << "Failed call to clGetContextInfo(...,CL_CONTEXT_DEVICES,...)";
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

    // Just choose the first available device
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

cl_program CreateProgram(cl_context context, cl_device_id device,
                         const char *fileName) {
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

// This program performs a parallel Bubble Sort on
// a list passed via the command line or randomly generated
int main(int argc, char *argv[]) {
    srand(time(0)); // Seed the random generator
    int list_size = DEFAULT_LIST_SIZE;
    cl_int errNum;

    // Check if list was passed via command line
    if (argc > 1) {
        list_size = argc - 1;
    }

    // Allocate space for the list
    int *list = (int *) malloc(list_size * sizeof(int));

    // Fill the list with given arguments, or make random ones
    for (int i = 0; i < list_size; i++) {
        if (argc > 1) {
            list[i] = atoi(argv[i + 1]);
        } else {
            list[i] = rand() % MAX_ELEMENT_SIZE;
        }
    }

    // Print the input list
    std::cout << "Bubble Sort -- Input List" << std::endl;
    printList(list, list_size);

    // Create OpenCL context
    cl_context context = CreateContext();
    if (context == NULL) {
        std::cerr << "Failed to create context." << std::endl;
        return 1;
    }

    // Create commandQueue and set device_id
    cl_device_id device;
    cl_command_queue commandQueue = CreateCommandQueue(context, &device);
    if (commandQueue == NULL) {
        std::cerr << "Failed to create commandQueue." << std::endl;
        return 1;
    }

    // Create program from assignment.cl kernel source
    cl_program program = CreateProgram(context, device, "assignment.cl");
    if (program == NULL) {
        std::cerr << "Failed to create program." << std::endl;
        return 1;
    }

    // Create kernel object
    cl_kernel kernel = clCreateKernel(program, "bubble_sort", NULL);
    if (kernel == NULL) {
        std::cerr << "Failed to create kernel" << std::endl;
        return 1;
    }

    // Set main kernel arguments
    cl_mem deviceList = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * list_size, list,
                                       NULL);
    cl_int deviceListSize = list_size;

    errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), &deviceList);
    errNum |= clSetKernelArg(kernel, 1, sizeof(cl_int), &deviceListSize);
    if (errNum != CL_SUCCESS) {
        std::cerr << "Error setting kernel arguments." << std::endl;
        return 1;
    }

    size_t globalWorkSize[1] = {(unsigned long) list_size};
    size_t localWorkSize[1] = {1};

    // Setup timing and start timer
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;
    auto t_start = high_resolution_clock::now();

    // Sort the list while it is not sorted
    bool sorted = checkSorted(list, list_size);
    while (!sorted) {
        // Do even pass of the sorting
        cl_int deviceSortingParity = 0;
        errNum |= clSetKernelArg(kernel, 2, sizeof(cl_int), &deviceSortingParity);
        if (errNum != CL_SUCCESS) {
            std::cerr << "Error setting kernel arguments." << std::endl;
            return 1;
        }

				cl_event evenParityDone;

        // Queue the sorting kernel up for execution
        errNum = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL,
                                        globalWorkSize, localWorkSize,
                                        0, NULL, &evenParityDone);
        if (errNum != CL_SUCCESS) {
            std::cerr << "Error queuing kernel for execution." << std::endl;
            return 1;
        }

				// Wait for even pass to finish
				errNum = clWaitForEvents(1, &evenParityDone);
				if (errNum != CL_SUCCESS) {
            std::cerr << "Even parity pass encountered a fatal error." << std::endl;
            return 1;
        }

        // Do odd pass of the sorting
        deviceSortingParity = 1;
        errNum |= clSetKernelArg(kernel, 2, sizeof(cl_int), &deviceSortingParity);
        if (errNum != CL_SUCCESS) {
            std::cerr << "Error setting kernel arguments." << std::endl;
            return 1;
        }

				cl_event oddParityDone;

        // Queue the sorting kernel up for execution
        errNum = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL,
                                        globalWorkSize, localWorkSize,
                                        0, NULL, &oddParityDone);
        if (errNum != CL_SUCCESS) {
            std::cerr << "Error queuing kernel for execution." << std::endl;
            return 1;
        }

				// Wait for odd pass to finish
				errNum = clWaitForEvents(1, &oddParityDone);
				if (errNum != CL_SUCCESS) {
            std::cerr << "Odd parity pass encountered a fatal error." << std::endl;
            return 1;
        }

        // Read the output back to the host
				cl_event readDone;
        errNum = clEnqueueReadBuffer(commandQueue, deviceList, CL_TRUE,
                                     0, list_size * sizeof(int), list,
                                     0, NULL, &readDone);
        if (errNum != CL_SUCCESS) {
            std::cerr << "Error reading result buffer." << std::endl;
            return 1;
        }

				// Wait for read from device to finish
				errNum = clWaitForEvents(1, &readDone);
				if (errNum != CL_SUCCESS) {
						std::cerr << "Reading encountered a fatal error." << std::endl;
						return 1;
				}

        // Check if sorted
        sorted = checkSorted(list, list_size);
    }

    // Stop the timer
    auto t_stop = high_resolution_clock::now();
    duration<double, std::milli> ms_double = t_stop - t_start;
    double total_time = ms_double.count();

    // Print the sorted list
    std::cout << "Sorted List" << std::endl;
    printList(list, list_size);

    // Print timing information
    std::cout << "Sorting took " << total_time << "(ms)..." << std::endl;

    // Cleanup
    clReleaseMemObject(deviceList);
    clReleaseCommandQueue(commandQueue);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseContext(context);
    free(list);

    return 0;
}
