#include "cl_helpers.h"
#include <cstdlib>
#include <math.h>
#include <random>
#include <time.h>
#include <stdio.h>
#include <string>

using namespace std;

cl_context initialize_opencl()
{
    cl_int error;

    vector<cl_platform_id> platform_ids = GetPlatformIDs();
	cout << "Number of platforms: " << (int)platform_ids.size()<< endl;

    cl_platform_id pid = platform_ids[0];

    vector<cl_device_id> device_ids =  GetDeviceIDs(pid);
	cout << "Number of devices: " << (int)device_ids.size()<< endl;
    for(int i = 0; i < (int)device_ids.size(); i++)
    {
        cout << GetDeviceName(device_ids[i]) << endl;
    }

    const cl_context_properties contextProperties [] =
    {
        CL_CONTEXT_PLATFORM,
        reinterpret_cast<cl_context_properties> (pid),
        0, 0
    };

    cl_context context = clCreateContext (
        contextProperties, device_ids.size(),
        device_ids.data (), nullptr,
        nullptr, &error); 
    CheckError(error);

    return context;
}

void readParams(string filename, int* numpts, double* box_size_x, double* box_size_y, double* density, double* viscosity, double* velocity, double* time, int* local_size)
{

    std::ifstream f(filename);
    string line;
    string params[] = {"number of particles : ", "box size x : ", "box size y : ",
                       "density : ", "viscosity : ", "velocity : ", "simulation time : ",
                       "work group size : " };
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
             if (line.find(params[6]) != string::npos) *local_size = (int)strtof(line.substr(params[7].size()).c_str(), &pEnd);

        }
        f.close();
    }
    else cout << "Unable to open parameter file \n";
}

int main()
{

    int numpts, local_size;
    double box_size_x, box_size_y, density, viscosity, velocity, time;
    cl_int error;
    string filename = "SPH.params";

    readParams(filename, &numpts, &box_size_x, &box_size_y,
               &density, &viscosity, &velocity, &time, &local_size);
    /* cout << time << endl; */

    const double dx = box_size_x / numpts;
    int num_work_groups = (numpts + local_size -1) / local_size;
    int gwsize = numpts + local_size - numpts % local_size;
	const size_t global_work_size[] = { gwsize };
    const size_t local_work_size[] = { local_size };

    cout << "\n\nNumber of work groups: " << num_work_groups << endl;
    cout << "Number of points: " << numpts << endl;

    vector<cl_float2> x(numpts);

    cl_context context = initialize_opencl();

    cl_float2 f = (cl_float2){1., 2.};

    for(int i=0; i<numpts; i++)
        x[i] = (cl_float2){i, i};

    cl_mem buf_x;
	buf_x = clCreateBuffer(context,
                          CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                          sizeof(cl_float2)*numpts, x.data(),
                          &error);
	CheckError(error);
}
