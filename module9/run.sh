make thrustAssignment

echo -e "\n\nTHRUST DEMO 0\n"
./thrustAssignment

echo -e "\n\nTHRUST DEMO 1\n"
./thrustAassignment 512 512

echo -e "\n\nTHRUST DEMO 2\n"
./assignment 524288 32


make nppNvgraphAssignment

echo -e "\n\nNPP/NVGRAPH DEMO 0\n"
./nppNvgraphAssignment


echo -e "\n\nNPP/NVGRAPH DEMO 1\n"
./nppNvgraphAssignment lena.pgm 128

echo -e "\n\nNPP/NVGRAPH DEMO 2\n"
./nppNvgraphAssignment lena.pgm 128
