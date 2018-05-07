All the codes have been tested on system with following:-

OS: Ubuntu 17.04, MacOS High Sierra 10.13.4;
Compiler: gcc 6.4.0;
Programming Language: c++11;
GPU: NVIDIA GeForce GTX 1050 Ti, AMD Radeon Pro 450 Compute Engine
Parallel Framework: OpenCL


Run the following commands to simulate the example case of flow past airfoil. Explanation follows

python geometry.py
g++ main.cpp -framework opencl
./a.out SPH.params
python animation.py output airfoil_wall.csv

Generating geometry:
This step requires pysph python package
Change output file name and geometry type if required.

python geometry.py

Parameter File:

SPH.params contains the relevant input parameters.
The parameter names are self explanatory.


Compilation:

On MacOS:
g++ main.cpp -framework OpenCl

On Ubuntu:
g++ main.cpp -lOpenCL

Run:

./a.out SPH.params

Animation:

output: The directory in which the output files of simulation are saved 
airfoil_wall.csv: The csv file containing the wall geometry

python animation.py output airfoil_wall.csv

 



