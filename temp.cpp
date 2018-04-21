#include "cl_helpers.h"
#include <cstdlib>
#include <math.h>
#include <random>
#include <time.h>
#include <stdio.h>

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


int main()
{

    cl_int error;
    int numpts = 256, local_size = 32;
    double range = 1;

    const double dx = range / numpts;
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
