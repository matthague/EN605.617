CC = g++

all: clean assignment

assignment: assignment.cpp
	$(CC) -std=c++11 -I/usr/local/cuda-8.0/targets/x86_64-linux/include/ -Icommon/inc  assignment.cpp -lOpenCL -o assignment

clean:
	rm -f *.o
