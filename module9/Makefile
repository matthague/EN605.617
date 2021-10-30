CC = nvcc

all: clean thrustAssignment nppNvgraphAssignment

assignment: thrustAssignment.cu
	$(CC) $^ -o $@ -lcudart

nppNvgraphAssignment: nppNvgraphAssignment.cu
	$(CC) -I./common/UtilNPP -I./common/FreeImage/include $^ -o $@ -lcudart -lfreeimage -lnppi_static -lnppc_static -lculibos -lfreeimage

clean:
	rm -f *.o