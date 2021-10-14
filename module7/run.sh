cmake .
make -j

./assignment 1 1 1 1

echo -e "\n\nDEMO 0\n"
./assignment

echo -e "\n\nDEMO 1\n"
./assignment 65536 256 64

echo -e "\n\nDEMO 2\n"
./assignment 1048576 256 1

echo -e "\n\nDEMO 3\n"
./assignment 1048576 1 64


echo -e "\n\nDEMO 4\n"
./assignment 16777216 4 64
