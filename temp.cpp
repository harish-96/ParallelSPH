#include "cl_helpers.h"
#include <cstdlib>
#include <math.h>
#include <random>
#include <time.h>
#include <stdio.h>

const unsigned char TRAPZ_FLAG = 0x1; // hex for 0000 0001 
const unsigned char MC_FLAG = 0x2; // hex for 0000 0010

using namespace std;


double Integral(int numpts, int local_size, double range, cl_device_id did,cl_context context, cl_kernel kernel, cl_kernel kernel_red, cl_kernel kernel_mc, unsigned char method_flag=TRAPZ_FLAG)
{
    cl_int error;
    cl_event event, event_red;
    const double dx = range / numpts;
    int num_work_groups = (numpts + local_size -1) / local_size;
    int gwsize = numpts + local_size - numpts % local_size;
	const size_t global_work_size[] = { gwsize };
    const size_t local_work_size[] = { local_size };
    const size_t global_work_size_red[] = { 256 };
    const size_t local_work_size_red[] = { 256 };

    cout << "\n\nNumber of work groups: " << num_work_groups << endl;
    cout << "Number of points: " << numpts << endl;

    vector<double> blk_sum(num_work_groups);
    vector<double> x(gwsize);

    double lower_bound = 0;
    double upper_bound = M_PI;
    std::uniform_real_distribution<double> unif(lower_bound,upper_bound);
    std::default_random_engine re;
    re.seed(time(NULL));

    for (int i=0; i < num_work_groups; i++) {
		blk_sum[i] = static_cast<double>(0);
	}
    
    if(method_flag & MC_FLAG){
        for (int i=0; i < gwsize; i++){
            if (i < numpts) x[i] = unif(re);
            else x[i] = 0;
        }
    }

    cl_mem buf_blk_sum, buf_x, buf_red;

	buf_blk_sum = clCreateBuffer(context,
                              CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                              sizeof(double) * blk_sum.size(), blk_sum.data(),
                              &error);
	CheckError(error);
    
    if(method_flag & MC_FLAG)
	buf_x = clCreateBuffer(context,
                              CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                              sizeof(double) * x.size(), x.data(),
                              &error);
	CheckError(error);

    double red = 0;
	buf_red = clCreateBuffer(context,
                              CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                              sizeof(double), &red,
                              &error);
	CheckError(error);

    if(method_flag & TRAPZ_FLAG){
	clSetKernelArg(kernel, 0, sizeof(double), &dx);
	clSetKernelArg(kernel, 1, sizeof(int), &numpts);
	clSetKernelArg(kernel, 2, sizeof(double)*local_work_size[0], NULL);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &buf_blk_sum);
    }

    if(method_flag & MC_FLAG){
	clSetKernelArg(kernel_mc, 0, sizeof(cl_mem), &buf_x);
	clSetKernelArg(kernel_mc, 1, sizeof(double)*local_work_size[0], NULL);
    clSetKernelArg(kernel_mc, 2, sizeof(cl_mem), &buf_blk_sum);
    }

	clSetKernelArg(kernel_red, 0, sizeof(cl_mem), &buf_blk_sum);
	clSetKernelArg(kernel_red, 1, sizeof(cl_mem), &buf_red);
	clSetKernelArg(kernel_red, 2, sizeof(double)*local_work_size_red[0], NULL);
    clSetKernelArg(kernel_red, 3, sizeof(int), &num_work_groups);

	cl_command_queue Q = clCreateCommandQueue(context, did,
	                                          CL_QUEUE_PROFILING_ENABLE, &error);
	CheckError(error);

    if(method_flag & TRAPZ_FLAG){
        cout << "\nTRAPEZOIDAL RULE:\n";
        CheckError(clEnqueueNDRangeKernel(Q, kernel, 1,
                                              NULL, global_work_size, local_work_size,
                                              0, nullptr, &event));
        CheckError(clEnqueueNDRangeKernel(Q, kernel_red, 1,
                                              NULL, global_work_size_red, local_work_size_red,
                                              0, nullptr, &event_red));

        clWaitForEvents(1, &event);
        clWaitForEvents(1, &event_red);
        cl_ulong time_start;
        cl_ulong time_end;

        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);

        double trapz_time = time_end-time_start;
        clGetEventProfilingInfo(event_red, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
        clGetEventProfilingInfo(event_red, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);

        trapz_time += time_end-time_start;
        printf("Execution time is: %0.3f milliseconds \n", trapz_time / 1000000.0);
        cout << trapz_time / 1000000 << "+";
        CheckError(clEnqueueReadBuffer(Q, buf_red, true, 0,
                                           sizeof(double), &red,
                                           0, nullptr, nullptr));

        cout << "Integral :  " << red * dx << endl;
        cout << "Error :  " << 2 - red * dx << endl;
    }
                
    if(method_flag & MC_FLAG){
        cout << "\nMONTE CARLO:\n";
        CheckError(clEnqueueNDRangeKernel(Q, kernel_mc, 1,
                                              NULL, global_work_size, local_work_size,
                                              0, nullptr, &event));
        CheckError(clEnqueueNDRangeKernel(Q, kernel_red, 1,
                                              NULL, global_work_size_red, NULL,
                                              0, nullptr, &event_red));

        clWaitForEvents(1, &event);
        clWaitForEvents(1, &event_red);
        cl_ulong time_start;
        cl_ulong time_end;

        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);

        double mc_time = time_end-time_start;
        clGetEventProfilingInfo(event_red, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
        clGetEventProfilingInfo(event_red, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);

        mc_time += time_end-time_start;
        printf("Execution time is: %0.3f milliseconds \n", mc_time / 1000000.0);

        CheckError(clEnqueueReadBuffer(Q, buf_red, true, 0,
                                           sizeof(double), &red,
                                           0, nullptr, nullptr));
        cout << "Integral :  " << red * dx << endl;
        cout << "Error :  " << 2 - red * dx << endl;
    }

	clReleaseCommandQueue(Q);
	
	clReleaseMemObject(buf_blk_sum);
	clReleaseMemObject(buf_red);
	if(method_flag & MC_FLAG) clReleaseMemObject(buf_x);

    return red*dx;

    }

int main()
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

    cl_program program = CreateProgram(file_to_string("trapz.cl"), context);
    clBuildProgram(program, device_ids.size(), device_ids.data(),
	                              nullptr, nullptr, nullptr);

    cl_kernel kernel = clCreateKernel(program, "TRAPZ", &error);
    CheckError(error);

    cl_kernel kernel_red = clCreateKernel(program, "REDUCE", &error);
    CheckError(error);

    cl_kernel kernel_mc = clCreateKernel(program, "MONTEC", &error);
    CheckError(error);


	const double range = M_PI;
    int local_work_size = 256;
    Integral(2048, local_work_size, range, device_ids[2], context, kernel, kernel_red, kernel_mc, MC_FLAG | TRAPZ_FLAG);

	clReleaseKernel(kernel);
	clReleaseKernel(kernel_mc);
	clReleaseKernel(kernel_red);
	clReleaseProgram(program);
	clReleaseContext(context);
	                                     
	return 0;

}
