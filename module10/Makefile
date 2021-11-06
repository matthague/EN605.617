CC = g++

all: clean assignment

assignment: assignment.cpp
	$(CC) -I/usr/local/cuda-8.0/targets/x86_64-linux/include assignment.cpp -lOpenCL -o assignment

clean:
	rm -f *.o
