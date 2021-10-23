CC = nvcc

all: clean assignment

assignment: assignment.cu
	$(CC) $^ -o $@ -lcudart -lcublas -lcusolver

clean:
	rm -f *.o