#include "cl_helpers.h"
#include <cstdlib>
#include <math.h>
#include <random>
#include <time.h>
#include <stdio.h>

using namespace std;

void readParams(string filename, int* numpts, double* box_size_x, double* box_size_y, double* density, double* viscosity, double* velocity, double* time, int* local_size, double *dt)
{

    int nf = 7; //Number of floating point parameters
    int nd = 2; //Number of integer parameters
    
    int *d_vars[] = { numpts, local_size };
    double *f_vars[] = { box_size_x,  box_size_y, density, viscosity, velocity, time, dt };

    string d_params[] = {"number of particles : ", "threads per block : "};
    string f_params[] = {"box size x : ", "box size y : ", "density : ",
                         "viscosity : ", "velocity : ", "simulation time : ",
                         "time step : " };

    ifstream f(filename);
    string line;

    if(f.is_open())
    {
        while(getline (f,line) )
        {
             int found;
             char* pEnd;
             transform(line.begin(), line.end(), line.begin(), ::tolower);

             for (int i=0; i < nd; i++){
                 if (line.find(d_params[i]) != string::npos)
                     *(d_vars[i]) = (int)strtof(line.substr(d_params[i].size()).c_str(), &pEnd);
             }
             for (int i=0; i < nf; i++){
                 if (line.find(f_params[i]) != string::npos)
                     *(f_vars[i]) = strtof(line.substr(f_params[i].size()).c_str(), &pEnd);
             }

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

    vector<cl_float2> x(numpts), v(numpts), r(numpts);
    set_ic(x, v, r);

    cl_context context;
    cl_device_id did;
    initialize_opencl(&context, &did);

    cl_mem buf_x, buf_r, buf_v;

	buf_x = clCreateBuffer(context,
                          CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                          sizeof(cl_float2)*numpts, x.data(),
                          &error);
	CheckError(error);
    
	buf_r = clCreateBuffer(context,
                          CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                          sizeof(cl_float2)*numpts, r.data(),
                          &error);
	CheckError(error);
    
	buf_v = clCreateBuffer(context,
                          CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                          sizeof(cl_float2)*numpts, v.data(),
                          &error);
	CheckError(error);
    
    /* CREATE KERNEL, SET ARGS, CREATE COMMAND QUEUE, LAUNCH KERNELS */
    cl_program program = CreateProgram(file_to_string("kernels.cl"), context);
    cl_device_id build_list[] = { did };
    clBuildProgram(program, 1, build_list,
                   nullptr, nullptr, nullptr);
    cl_kernel kernel = clCreateKernel(program, "SUMDEN", &error);
    CheckError(error);

}
