CC = nvcc

all: clean thrustAssignment nppNvgraphAssignment

thrustAssignment: thrustAssignment.cu
	$(CC) $^ -o $@ -lcudart

nppNvgraphAssignment: nppNvgraphAssignment.cu
	$(CC) -I./common/UtilNPP -I./common/FreeImage/include $^ -o $@  -L./common/FreeImage/lib/ -L./common/FreeImage/lib/linux -L./common/FreeImage/lib/linux/x86_64 -lcudart -lnppi_static -lnppc_static -lculibos -lnvgraph

clean:
	rm -f *.o
