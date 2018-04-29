#include "cl_helpers.h"
#include <cstdlib>
#include <math.h>
#include <random>
#include <time.h>
#include <stdio.h>

using namespace std;

void readParams(string filename, int* numpts, int*Nw, float* box_size_x, float* box_size_y, float* density, float* viscosity, float* velocity, float* time, int* local_size, float *dt, float *h)
{

    int nf = 8; //Number of floating point parameters
    int nd = 3; //Number of integer parameters
    
    int *d_vars[] = { numpts, Nw, local_size };
    float *f_vars[] = { box_size_x,  box_size_y, density, viscosity, velocity, time, dt, h };

    string d_params[] = {"number of fluid particles : ", "number of wall particles : ","threads per block : "};
    string f_params[] = {"box size x : ", "box size y : ", "density : ",
                         "viscosity : ", "velocity : ", "simulation time : ",
                         "time step : ", "kernel size : "};

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

void set_ic(vector<cl_float2> &x, vector<cl_float2> &xw, vector<cl_float2> &v, vector<cl_float2> &vw, vector<cl_float> &r, vector<cl_float> &rw, float dx)
{
    for (int i=0; i < x.size(); i++)
    {
        x[i].s[0] = i * dx;
        x[i].s[1] = i * dx;
        v[i].s[0] = 1;
        v[i].s[1] = 0;
        r[i] = 1000;
    }
    for (int i=0; i < x.size(); i++)
        cout << "Particle number" << i << "\t" << x[i].s[0] << "\t" << x[i].s[1] << endl;
    for (int i=0; i < xw.size(); i++)
    {
        xw[i].s[0] = 1000;
        xw[i].s[1] = 1000;
        vw[i].s[0] = 1;
        vw[i].s[1] = 0;
        rw[i] = 100000;
    }
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
    int numpts, Nw, local_size;
    float box_size_x, box_size_y, rho0, viscosity, velocity, time, dt, h;

    readParams(input_params, &numpts, &Nw, &box_size_x, &box_size_y,
               &rho0, &viscosity, &velocity, &time,
               &local_size, &dt, &h);


    float c0 = 10;
    float dx = h/1.1, m = rho0 * dx * dx;

    int num_work_groups = (numpts + local_size -1) / local_size;
    int gwsize = numpts + local_size - numpts % local_size;
	const size_t global_work_size[] = { gwsize };
    const size_t local_work_size[] = { local_size };

    int num_work_groups_w = (Nw + local_size -1) / local_size;
    int gwsize_w = Nw + local_size - Nw % local_size;
	const size_t global_work_size_w[] = { gwsize_w };

    vector<cl_float2> x(numpts), xw(Nw), vw(Nw), v(numpts);
    vector<cl_float> r(numpts), rw(Nw), p(numpts), pw(Nw);
    set_ic(x, xw, v, vw, r, rw, dx);

    cl_context context;
    cl_device_id did;
    initialize_opencl(&context, &did);

    cl_mem buf_x, buf_xw, buf_r, buf_rw, buf_v, buf_vw, buf_p, buf_pw;

	buf_x = clCreateBuffer(context,
                          CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                          sizeof(cl_float2)*numpts, x.data(),
                          &error);
	CheckError(error);
    
	buf_xw = clCreateBuffer(context,
                           CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                           sizeof(cl_float2)*Nw, xw.data(),
                           &error);
	CheckError(error);

	buf_r = clCreateBuffer(context,
                          CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                          sizeof(cl_float)*numpts, r.data(),
                          &error);
	CheckError(error);

	buf_rw = clCreateBuffer(context,
                          CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                          sizeof(cl_float)*Nw, rw.data(),
                          &error);
	CheckError(error);
    
	buf_p = clCreateBuffer(context,
                          CL_MEM_READ_WRITE,
                          sizeof(cl_float)*numpts, NULL,
                          &error);
	CheckError(error);

	buf_pw = clCreateBuffer(context,
                          CL_MEM_READ_WRITE,
                          sizeof(cl_float)*Nw, NULL,
                          &error);
	CheckError(error);

	buf_v = clCreateBuffer(context,
                          CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                          sizeof(cl_float2)*numpts, v.data(),
                          &error);
	CheckError(error);
    
	buf_vw = clCreateBuffer(context,
                          CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                          sizeof(cl_float2)*Nw, vw.data(),
                          &error);
	CheckError(error);

    /* BUILD PROGRAM, CREATE KERNEL, SET ARGS, CREATE COMMAND QUEUE, LAUNCH KERNELS */
    char options[] = "-g";
    cl_program program = CreateProgram(file_to_string("kernels.cl"), context);
    cl_device_id build_list[] = { did };
    clBuildProgram(program, 1, build_list,
                   options, nullptr, nullptr);

    cl_kernel kernel_den = clCreateKernel(program, "DEN", &error);
    CheckError(error);

    cl_kernel kernel_pos = clCreateKernel(program, "UPDATE_POS", &error);
    CheckError(error);

    cl_kernel kernel_p = clCreateKernel(program, "INCOMP_P", &error);
    CheckError(error);

    cl_kernel kernel_vel = clCreateKernel(program, "UPDATE_VEL", &error);
    CheckError(error);

    cl_kernel kernel_wall = clCreateKernel(program, "WALL", &error);
    CheckError(error);

	clSetKernelArg(kernel_den, 0, sizeof(cl_mem), &buf_x);
	clSetKernelArg(kernel_den, 1, sizeof(cl_mem), &buf_xw);
	clSetKernelArg(kernel_den, 2, sizeof(cl_mem), &buf_v);
	clSetKernelArg(kernel_den, 3, sizeof(cl_mem), &buf_vw);
	clSetKernelArg(kernel_den, 4, sizeof(cl_mem), &buf_r);
	clSetKernelArg(kernel_den, 5, sizeof(cl_mem), &buf_rw);
	clSetKernelArg(kernel_den, 6, sizeof(float), &m);
	clSetKernelArg(kernel_den, 7, sizeof(float), &h);
	clSetKernelArg(kernel_den, 8, sizeof(float), &dt);
	clSetKernelArg(kernel_den, 9, sizeof(int), &numpts);
	clSetKernelArg(kernel_den, 10, sizeof(int), &Nw);

	clSetKernelArg(kernel_pos, 0, sizeof(cl_mem), &buf_x);
	clSetKernelArg(kernel_pos, 1, sizeof(cl_mem), &buf_v);
	clSetKernelArg(kernel_pos, 2, sizeof(cl_mem), &buf_r);
	clSetKernelArg(kernel_pos, 3, sizeof(float), &m);
	clSetKernelArg(kernel_pos, 4, sizeof(float), &h);
	clSetKernelArg(kernel_pos, 5, sizeof(int), &numpts);
	clSetKernelArg(kernel_pos, 6, sizeof(float), &dt);

	clSetKernelArg(kernel_p, 0, sizeof(cl_mem), &buf_r);
	clSetKernelArg(kernel_p, 1, sizeof(cl_mem), &buf_p);
	clSetKernelArg(kernel_p, 2, sizeof(float), &c0);
	clSetKernelArg(kernel_p, 3, sizeof(float), &rho0);

	clSetKernelArg(kernel_vel, 0, sizeof(cl_mem), &buf_x);
	clSetKernelArg(kernel_vel, 1, sizeof(cl_mem), &buf_xw);
	clSetKernelArg(kernel_vel, 2, sizeof(cl_mem), &buf_p);
	clSetKernelArg(kernel_vel, 3, sizeof(cl_mem), &buf_pw);
	clSetKernelArg(kernel_vel, 4, sizeof(cl_mem), &buf_v);
	clSetKernelArg(kernel_vel, 5, sizeof(cl_mem), &buf_vw);
	clSetKernelArg(kernel_vel, 6, sizeof(cl_mem), &buf_r);
	clSetKernelArg(kernel_vel, 7, sizeof(cl_mem), &buf_rw);
	clSetKernelArg(kernel_vel, 8, sizeof(float), &m);
	clSetKernelArg(kernel_vel, 9, sizeof(int), &numpts);
	clSetKernelArg(kernel_vel, 10, sizeof(int), &Nw);
	clSetKernelArg(kernel_vel, 11, sizeof(float), &dt);
	clSetKernelArg(kernel_vel, 12, sizeof(float), &h);

	clSetKernelArg(kernel_wall, 0, sizeof(cl_mem), &buf_x);
	clSetKernelArg(kernel_wall, 1, sizeof(cl_mem), &buf_v);
	clSetKernelArg(kernel_wall, 2, sizeof(cl_mem), &buf_vw);
	clSetKernelArg(kernel_wall, 3, sizeof(cl_mem), &buf_p);
	clSetKernelArg(kernel_wall, 4, sizeof(cl_mem), &buf_pw);
	clSetKernelArg(kernel_wall, 5, sizeof(cl_mem), &buf_rw);
	clSetKernelArg(kernel_wall, 6, sizeof(float), &h);
	clSetKernelArg(kernel_wall, 7, sizeof(float), &rho0);
	clSetKernelArg(kernel_wall, 8, sizeof(float), &c0);
	clSetKernelArg(kernel_wall, 9, sizeof(int), &numpts);

	cl_command_queue Q = clCreateCommandQueue(context, did,
	                                          CL_QUEUE_PROFILING_ENABLE, &error);
    CheckError(error);
    cl_event event;
    CheckError(clEnqueueNDRangeKernel(Q, kernel_den, 1,
                                      NULL, global_work_size, local_work_size,
                                      0, NULL, &event));
    CheckError(clEnqueueNDRangeKernel(Q, kernel_p, 1,
                                      NULL, global_work_size, local_work_size,
                                      0, NULL, &event));
    /* CheckError(clEnqueueNDRangeKernel(Q, kernel_wall, 1, */
    /*                                   NULL, global_work_size, local_work_size, */
    /*                                   0, NULL, &event)); */
    /* CheckError(clEnqueueNDRangeKernel(Q, kernel_vel, 1, */
    /*                                   NULL, global_work_size, local_work_size, */
    /*                                   0, NULL, &event)); */
    /* CheckError(clEnqueueNDRangeKernel(Q, kernel_pos, 1, */
    /*                                   NULL, global_work_size, local_work_size, */
    /*                                   0, NULL, &event)); */

    CheckError(clEnqueueReadBuffer(Q, buf_p, true, 0,
                                       sizeof(cl_float)*numpts, p.data(),
                                       0, nullptr, nullptr));
    CheckError(clEnqueueReadBuffer(Q, buf_r, true, 0,
                                       sizeof(cl_float)*numpts, r.data(),
                                       0, nullptr, nullptr));
    CheckError(clEnqueueReadBuffer(Q, buf_v, true, 0,
                                       sizeof(cl_float2)*numpts, v.data(),
                                       0, nullptr, nullptr));
    CheckError(clEnqueueReadBuffer(Q, buf_x, true, 0,
                                       sizeof(cl_float2)*numpts, x.data(),
                                       0, nullptr, nullptr));
    for (int i=0; i < x.size(); i++)
        cout << "Particle number" << i << "\t" << r[i] << "\t" << p[i] << endl;
}
