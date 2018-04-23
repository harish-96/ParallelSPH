#include "cl_helpers.h"
#include <cstdlib>
#include <math.h>
#include <random>
#include <time.h>
#include <stdio.h>

using namespace std;

void readParams(string filename, int* numpts, double* box_size_x, double* box_size_y, double* density, double* viscosity, double* velocity, double* time, int* local_size, double *dt)
{

    std::ifstream f(filename);
    string line;
    string params[] = {"number of particles : ", "box size x : ", "box size y : ",
                       "density : ", "viscosity : ", "velocity : ", "simulation time : ",
                       "threads per block : ", "time step : " };
    if(f.is_open())
    {
        while(getline (f,line) )
        {
             int found;
             char* pEnd;
             transform(line.begin(), line.end(), line.begin(), ::tolower);

             if (line.find(params[0]) != string::npos) *numpts = (int)strtof(line.substr(params[0].size()).c_str(), &pEnd);
             if (line.find(params[1]) != string::npos) *box_size_x = strtof(line.substr(params[1].size()).c_str(), &pEnd);
             if (line.find(params[2]) != string::npos) *box_size_y = strtof(line.substr(params[2].size()).c_str(), &pEnd);
             if (line.find(params[3]) != string::npos) *density = strtof(line.substr(params[3].size()).c_str(), &pEnd);
             if (line.find(params[4]) != string::npos) *viscosity = strtof(line.substr(params[4].size()).c_str(), &pEnd);
             if (line.find(params[5]) != string::npos) *velocity = strtof(line.substr(params[5].size()).c_str(), &pEnd);
             if (line.find(params[6]) != string::npos) *time = strtof(line.substr(params[6].size()).c_str(), &pEnd);
             if (line.find(params[7]) != string::npos) *local_size = (int)strtof(line.substr(params[7].size()).c_str(), &pEnd);
             if (line.find(params[8]) != string::npos) *dt = strtof(line.substr(params[8].size()).c_str(), &pEnd);

        }
        f.close();
    }
    else
    {
        cout << filename << " does not exist. Exiting\n";
        exit(1);
    }
}

void set_ic(vector<cl_float2> x, vector<cl_float2> u, vector<cl_float2> r)
{

}

int main(int argc, char *argv[])
{
    string input_params;
    if(argc > 1) input_params = argv[1];
    else
    {
        cout << "Pass parameter file name as command line argument. Exiting\n";
        exit(1);
    }
    cl_int error;
    int numpts, local_size;
    double box_size_x, box_size_y, density, viscosity, velocity, time, dt;


    readParams(input_params, &numpts, &box_size_x, &box_size_y,
               &density, &viscosity, &velocity, &time,
               &local_size, &dt);
    /* cout << dt << endl; */

    const double dx = box_size_x / numpts;
    int num_work_groups = (numpts + local_size -1) / local_size;
    int gwsize = numpts + local_size - numpts % local_size;
	const size_t global_work_size[] = { gwsize };
    const size_t local_work_size[] = { local_size };

    vector<cl_float2> x(numpts), u(numpts), r(numpts);
    set_ic(x, u, r);

    cl_context context;
    cl_device_id did;
    initialize_opencl(&context, &did);

    cl_mem buf_x, buf_r, buf_u, buf_v;

	buf_x = clCreateBuffer(context,
                          CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                          sizeof(cl_float2)*numpts, x.data(),
                          &error);
	CheckError(error);
}
