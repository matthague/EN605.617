#include <ImagesCPU.h>
#include <ImagesNPP.h>
#include <ImageIO.h>
#include <Exceptions.h>

#include <string.h>
#include <fstream>
#include <iostream>

#include <cuda_runtime.h>
#include <npp.h>
#include <nppi.h>

#include <nvgraph.h>

#define DEFAULT_GRAPH_SIZE 20

using namespace std;

void printUsage(char *argv[]) {
    printf("Usage: %s <imagefile: REQUIRED> <graphsize: OPTIONAL>", argv[0]);
}

float doNPPBorderFinding(string inputFilename) {
    // setup timing
    float kernelTime;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // filenames
    string outputFilename = "output.pgm";

    // load gray-scale image from filename
    npp::ImageCPU_8u_C1 hostImage;
    npp::loadImage(inputFilename, hostImage);

    // copy host image to device
    npp::ImageNPP_8u_C1 deviceImage(hostImage);

    // find size
    NppiSize deviceImageSize = {(int) deviceImage.width(), (int) deviceImage.height()};
    NppiPoint deviceImageOffset = {0, 0};

    // allocate result device image
    npp::ImageNPP_8u_C1 deviceResult(deviceImageSize.width, deviceImageSize.height);

    // allocate scratch space
    int scratchBufferSize = 0;
    Npp8u *scratchBuffer = 0;
    nppiFilterCannyBorderGetBufferSize(deviceImageSize, &scratchBufferSize);
    cudaMalloc((void **) &scratchBuffer, scratchBufferSize);

    // run the edge detection filter (settings thanks to nvidia sample)
    Npp16s lowThreshold = 72;
    Npp16s highThreshold = 256;

    nppiFilterCannyBorder_8u_C1R(deviceImage.data(), deviceImage.pitch(), deviceImageSize, deviceImageOffset,
                                 deviceResult.data(), deviceResult.pitch(), deviceImageSize, NPP_FILTER_SOBEL,
                                 NPP_MASK_SIZE_3_X_3, lowThreshold, highThreshold, nppiNormL2, NPP_BORDER_REPLICATE,
                                 scratchBuffer);

    // copy back image and save
    npp::ImageCPU_8u_C1 hostResult(deviceResult.size());
    deviceResult.copyTo(hostResult.data(), hostResult.pitch());
    saveImage(outputFilename, hostResult);

    // free resources
    cudaFree(scratchBuffer);
    nppiFree(deviceImage.data());
    nppiFree(deviceResult.data());

    // stop the clock
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&kernelTime, start, stop);

    // return time
    return kernelTime;
}

float doNvgraphCountTriangles(int graphSize) {
    // setup handles and space for graph structure using COO topology
    nvgraphHandle_t handle;
    nvgraphGraphDescr_t graph;
    nvgraphCOOTopology32I_st* COO_input = (nvgraphCOOTopology32I_st*) malloc(sizeof(struct nvgraphCOOTopology32I_st));

    // init nvgraph hande and descriptor
    nvgraphCreate(&handle);
    nvgraphCreateGraphDescr (handle, &graph);

    // setup the graph edges and vertices
    const size_t numVertices = graphSize;
    const size_t numEdges = ((graphSize * (graphSize + 1))/2) - graphSize;
    int source_indices[numEdges];
    int destination_indices[numEdges];

    int ctr = 0;
    for(int i = 0; i < graphSize; i++) {
        for(int j = 0; j < i; j++) {
            source_indices[ctr] = i;
            destination_indices[ctr] = j;
            ctr += 1;
        }
    }

    COO_input->nvertices = numVertices;
    COO_input->nedges = numEdges;
    COO_input->source_indices = source_indices;
    COO_input->destination_indices = destination_indices;

    nvgraphSetGraphStructure(handle, graph, (void*)COO_input, NVGRAPH_COO_32);

    // setup timing
    float kernelTime;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // count the triangles
    uint64_t triangleCount;
    nvgraphTriangleCount(handle, graph, &triangleCount);

    // stop the clock
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&kernelTime, start, stop);

    // free resources
    free(COO_input);
    nvgraphDestroyGraphDescr(handle, graph);
    nvgraphDestroy(handle);

    // return time
    return kernelTime;
}

int main(int argc, char *argv[]) {
    // parse args
    if (argc > 3) {
        printUsage(argv);
    }

    string imageFile = "lena.pgm";
    if(argc > 1) {
        imageFile = string(argv[1]);
    }

    int graphSize = DEFAULT_GRAPH_SIZE;
    if(argc > 2) {
        graphSize = atoi(argv[2]);
    }

    // run each method and time
    float triangleCountingTime = doNvgraphCountTriangles(graphSize);
    float borderDetectionTime = doNPPBorderFinding(imageFile);

    // print results
    printf("Using Graph Size %d...\n", graphSize);
    printf("Triangle Counting Using NVGRAPH... : %f (ms)\n", triangleCountingTime);
    printf("Border Detection Using NPP... : %f (ms)\n", borderDetectionTime);

    return 0;
}

