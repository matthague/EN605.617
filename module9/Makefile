CC = nvcc

all: clean thrustAssignment nppNvgraphAssignment

thrustAssignment: thrustAssignment.cu
	$(CC) $^ -o $@ -lcudart

nppNvgraphAssignment: nppNvgraphAssignment.cu
	$(CC) -I./common/UtilNPP -I./common/FreeImage/include $^ -o $@ -lcudart -lnppi_static -lnppc_static -lculibos -lnvgraph

clean:
	rm -f *.o
