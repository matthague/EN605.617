cmake .
make -j

./assignment 1 1 1 1

echo -e "\n\nDEMO 0\n"
./assignment

echo -e "\n\nDEMO 1\n"
./assignment 1024 1024 8192

echo -e "\n\nDEMO 2\n"
./assignment 256 256 1

echo -e "\n\nDEMO 3\n"
./assignment 65536 1024 128

echo -e "\n\nDEMO 4\n"
./assignment 65536 256 4096
