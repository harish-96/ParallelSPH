#include "cl_helpers.h"
#include <cstdlib>
#include <math.h>
#include <random>
#include <time.h>
#include <stdio.h>
#include <algorithm>
using namespace std;

void readParams(string filename, string &output_dir, string &geometry_file, int* numpts, float* box_size_x, float* box_size_y, float* density, float* velocity, float* total_t, int* local_size, float *dt, float *h, int* saveFreq)
{

    int nf = 7; //Number of floating point parameters
    int nd = 3; //Number of integer parameters
    
    int *d_vars[] = { numpts, local_size, saveFreq };
    float *f_vars[] = { box_size_x,  box_size_y, density, velocity, total_t, dt, h };

    string d_params[] = {"number of fluid particles : ", "threads per block : ",
                         "checkpoint save frequency : "};
    string f_params[] = {"box size x : ", "box size y : ", "density : ",
                         "velocity : ", "simulation time : ",
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
             if (line.find("output directory") != string::npos)
                 output_dir = line.substr(19);
             if (line.find("geometry file : ") != string::npos)
                 geometry_file = line.substr(16);
        }

        f.close();
    }
    else
    {
        cout << filename << " does not exist. Exiting\n";
        exit(1);
    }
}

void set_ic(vector<cl_float2> &x, vector<cl_float2> &xw, vector<cl_float2> &v,
            vector<cl_float2> &vw, vector<cl_float> &r, vector<cl_float> &rw,
            float dx, float box_size_x, float box_size_y, float velocity,
            float density, string input_geometry)
{
    uniform_real_distribution<double> randx(0,box_size_x);
    uniform_real_distribution<double> randy(0,box_size_y);
    uniform_real_distribution<double> rvx(-velocity, velocity);
    uniform_real_distribution<double> rvy(-velocity, velocity);
    default_random_engine rex, rey, revx, revy;
    rex.seed(time(NULL));
    rey.seed(time(NULL));
    revx.seed(time(NULL));
    revy.seed(time(NULL));

    double dx1 = (double)box_size_x / pow(x.size(), 0.5);
    double dxw = (double)box_size_x / xw.size();
    for (int i=0; i < x.size(); i++)
    {
        x[i].s[0] = -1.1 + (i % (int)pow(x.size(), 0.5))*dx1;
        x[i].s[1] = ((i / (int)pow(x.size(), 0.5)))*dx1;
        v[i].s[0] = velocity;
        v[i].s[1] = 0;
        r[i] = density;
    }

    ifstream f(input_geometry);
    string line;

    if(f.is_open())
    {
        int i = 0;
        while(getline (f,line) )
        {
            int pos = line.find(",");
            char* pEnd;
            xw[i].s[0] = (float)strtof(line.substr(0, pos).c_str(), &pEnd);
            xw[i].s[1] = (float)strtof(line.substr(pos+1).c_str(), &pEnd);
            i++;
        }
    }
    for (int i=0; i < xw.size(); i++)
    {

        vw[i].s[0] = 0;
        vw[i].s[1] = 0;
        rw[i] = 100;
    }
}
void saveCheckpoint(string i, string output_dir, cl_mem &buf_x, cl_mem &buf_v, cl_mem &buf_vw, cl_mem &buf_r,cl_mem &buf_rw, cl_mem &buf_p,cl_mem &buf_pw, vector<cl_float2> &x, vector<cl_float2> &v, vector<cl_float> &r, vector<cl_float> &p, cl_command_queue &Q, int Nw)
{
    int numpts = x.size();
    string filename = output_dir + "/" + i + ".csv";
    ofstream f(filename); 
    f << "Particle Number,X pos,Y pos,X vel,Y vel,Density,Pressure\n";

    vector <float> pw(Nw), rw(Nw), d(Nw);
    vector <cl_float2> vw(Nw);
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

    CheckError(clEnqueueReadBuffer(Q, buf_vw, true, 0,
                                       sizeof(cl_float2)*Nw, vw.data(),
                                       0, nullptr, nullptr));
    CheckError(clEnqueueReadBuffer(Q, buf_pw, true, 0,
                                       sizeof(cl_float)*Nw, pw.data(),
                                       0, nullptr, nullptr));
    CheckError(clEnqueueReadBuffer(Q, buf_rw, true, 0,
                                       sizeof(cl_float)*Nw, rw.data(),
                                       0, nullptr, nullptr));
    
    for (int i=0; i < x.size(); i++)
    {
        f << i << "," << x[i].s[0] << "," << x[i].s[1] << "," << v[i].s[0];
        f << "," << v[i].s[1] << "," << r[i] << "," << p[i] << endl;
    }
    f.close();

    filename = "./wall/" + i + ".csv";
    ofstream f1(filename); 
    f1 << "Particle Number,X vel,Y vel,Density,Pressure,Dist\n";
    for (int i=0; i < rw.size(); i++)
    {
        f1 << i << "," << vw[i].s[0] << "," << vw[i].s[1] << ",";
        f1 << rw[i] << "," << pw[i] << endl;
    }
    f1.close();
}

int getNumLines(string geometry_file)
{
    int Nw = 0;
    string line;
    ifstream myfile(geometry_file);

    while (getline(myfile, line))
    {
        Nw++;
    }
    return Nw;
}

int main(int argc, char *argv[])
{
    string input_params, output_dir, geometry_file;
    if(argc > 1) input_params = argv[1];
    else
    {
        cout << "Pass parameter file name as command line argument. Exiting\n";
        exit(1);
    }
    cl_int error;
    int numpts, local_size, Nw, saveFreq;
    float box_size_x, box_size_y, rho0, viscosity, velocity, total_t, dt, h;

    readParams(input_params, output_dir, geometry_file, &numpts, &box_size_x, &box_size_y,
               &rho0, &velocity, &total_t,
               &local_size, &dt, &h, &saveFreq);

    Nw = getNumLines(geometry_file);

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
    set_ic(x, xw, v, vw, r, rw, dx, box_size_x, box_size_y,
           velocity, rho0, geometry_file);

    cl_context context;
    cl_device_id did;
    initialize_opencl(&context, &did);

    cl_mem buf_x, buf_xw, buf_r, buf_rw, buf_v, buf_tmpf,
           buf_vw, buf_p, buf_pw, buf_tmpf2, buf_d;

    buf_x = clCreateBuffer(context,
                          CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                          sizeof(cl_float2)*numpts, x.data(),
                          &error);
    CheckError(error);

    buf_tmpf2 = clCreateBuffer(context,
                          CL_MEM_READ_WRITE,
                          sizeof(cl_float2)*numpts, NULL,
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

    buf_tmpf = clCreateBuffer(context,
                          CL_MEM_READ_WRITE,
                          sizeof(cl_float)*numpts, NULL,
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
    cl_program program = CreateProgram(file_to_string("kernels.cl"), context);
    cl_device_id build_list[] = { did };
    clBuildProgram(program, 1, build_list,
                   NULL, NULL, NULL);

    cl_kernel kernel_den = clCreateKernel(program, "UPDATE_DEN", &error);
    CheckError(error);

    cl_kernel kernel_dr = clCreateKernel(program, "DELTA_DEN", &error);
    CheckError(error);

    cl_kernel kernel_pos = clCreateKernel(program, "UPDATE_POS", &error);
    CheckError(error);

    cl_kernel kernel_dx = clCreateKernel(program, "DELTA_X", &error);
    CheckError(error);

    cl_kernel kernel_p = clCreateKernel(program, "INCOMP_P", &error);
    CheckError(error);

    cl_kernel kernel_dv = clCreateKernel(program, "DELTA_V", &error);
    CheckError(error);

    cl_kernel kernel_vel = clCreateKernel(program, "UPDATE_VEL", &error);
    CheckError(error);

    cl_kernel kernel_wall = clCreateKernel(program, "WALL", &error);
    CheckError(error);

	clSetKernelArg(kernel_dr, 0, sizeof(cl_mem), &buf_x);
	clSetKernelArg(kernel_dr, 1, sizeof(cl_mem), &buf_xw);
	clSetKernelArg(kernel_dr, 2, sizeof(cl_mem), &buf_v);
	clSetKernelArg(kernel_dr, 3, sizeof(cl_mem), &buf_vw);
	clSetKernelArg(kernel_dr, 4, sizeof(cl_mem), &buf_r);
	clSetKernelArg(kernel_dr, 5, sizeof(cl_mem), &buf_rw);
	clSetKernelArg(kernel_dr, 6, sizeof(cl_mem), &buf_tmpf);
	clSetKernelArg(kernel_dr, 7, sizeof(float), &m);
	clSetKernelArg(kernel_dr, 8, sizeof(float), &h);
	clSetKernelArg(kernel_dr, 9, sizeof(float), &dt);
	clSetKernelArg(kernel_dr, 10, sizeof(int), &numpts);
	clSetKernelArg(kernel_dr, 11, sizeof(int), &Nw);

	clSetKernelArg(kernel_den, 0, sizeof(cl_mem), &buf_r);
	clSetKernelArg(kernel_den, 1, sizeof(cl_mem), &buf_tmpf);
	clSetKernelArg(kernel_den, 2, sizeof(int), &numpts);

	clSetKernelArg(kernel_dx, 0, sizeof(cl_mem), &buf_x);
	clSetKernelArg(kernel_dx, 1, sizeof(cl_mem), &buf_xw);
	clSetKernelArg(kernel_dx, 2, sizeof(cl_mem), &buf_tmpf2);
	clSetKernelArg(kernel_dx, 3, sizeof(cl_mem), &buf_v);
	clSetKernelArg(kernel_dx, 4, sizeof(cl_mem), &buf_vw);
	clSetKernelArg(kernel_dx, 5, sizeof(cl_mem), &buf_r);
	clSetKernelArg(kernel_dx, 6, sizeof(cl_mem), &buf_rw);
	clSetKernelArg(kernel_dx, 7, sizeof(float), &m);
	clSetKernelArg(kernel_dx, 8, sizeof(float), &h);
	clSetKernelArg(kernel_dx, 9, sizeof(int), &numpts);
	clSetKernelArg(kernel_dx, 10, sizeof(int), &Nw);
	clSetKernelArg(kernel_dx, 11, sizeof(float), &dt);

	clSetKernelArg(kernel_pos, 0, sizeof(cl_mem), &buf_x);
	clSetKernelArg(kernel_pos, 1, sizeof(cl_mem), &buf_tmpf2);
	clSetKernelArg(kernel_pos, 2, sizeof(int), &numpts);

	clSetKernelArg(kernel_p, 0, sizeof(cl_mem), &buf_r);
	clSetKernelArg(kernel_p, 1, sizeof(cl_mem), &buf_p);
	clSetKernelArg(kernel_p, 2, sizeof(float), &c0);
	clSetKernelArg(kernel_p, 3, sizeof(float), &rho0);
	clSetKernelArg(kernel_p, 4, sizeof(int), &numpts);

	clSetKernelArg(kernel_dv, 0, sizeof(cl_mem), &buf_x);
	clSetKernelArg(kernel_dv, 1, sizeof(cl_mem), &buf_xw);
	clSetKernelArg(kernel_dv, 2, sizeof(cl_mem), &buf_p);
	clSetKernelArg(kernel_dv, 3, sizeof(cl_mem), &buf_pw);
	clSetKernelArg(kernel_dv, 4, sizeof(cl_mem), &buf_v);
	clSetKernelArg(kernel_dv, 5, sizeof(cl_mem), &buf_vw);
	clSetKernelArg(kernel_dv, 6, sizeof(cl_mem), &buf_tmpf2);
	clSetKernelArg(kernel_dv, 7, sizeof(cl_mem), &buf_r);
	clSetKernelArg(kernel_dv, 8, sizeof(cl_mem), &buf_rw);
	clSetKernelArg(kernel_dv, 9, sizeof(float), &m);
	clSetKernelArg(kernel_dv, 10, sizeof(int), &numpts);
	clSetKernelArg(kernel_dv, 11, sizeof(int), &Nw);
	clSetKernelArg(kernel_dv, 12, sizeof(float), &dt);
	clSetKernelArg(kernel_dv, 13, sizeof(float), &h);

	clSetKernelArg(kernel_vel, 0, sizeof(cl_mem), &buf_v);
	clSetKernelArg(kernel_vel, 1, sizeof(cl_mem), &buf_tmpf2);
	clSetKernelArg(kernel_vel, 2, sizeof(int), &numpts);

	clSetKernelArg(kernel_wall, 0, sizeof(cl_mem), &buf_x);
	clSetKernelArg(kernel_wall, 1, sizeof(cl_mem), &buf_xw);
	clSetKernelArg(kernel_wall, 2, sizeof(cl_mem), &buf_v);
	clSetKernelArg(kernel_wall, 3, sizeof(cl_mem), &buf_vw);
	clSetKernelArg(kernel_wall, 4, sizeof(cl_mem), &buf_p);
	clSetKernelArg(kernel_wall, 5, sizeof(cl_mem), &buf_pw);
	clSetKernelArg(kernel_wall, 6, sizeof(cl_mem), &buf_rw);
	clSetKernelArg(kernel_wall, 7, sizeof(float), &h);
	clSetKernelArg(kernel_wall, 8, sizeof(float), &rho0);
	clSetKernelArg(kernel_wall, 9, sizeof(float), &c0);
	clSetKernelArg(kernel_wall, 10, sizeof(int), &numpts);
	clSetKernelArg(kernel_wall, 11, sizeof(int), &Nw);

	cl_command_queue Q = clCreateCommandQueue(context, did,
	                                          CL_QUEUE_PROFILING_ENABLE, &error);
    CheckError(error);
    cl_event event;

    string command_str = "mkdir -p " + output_dir;
    const char* command = command_str.c_str();
    system(command);

    command_str = "mkdir -p wall";
    const char* command2 = command_str.c_str();
    system(command2);

    int nTime = total_t / dt;
    CheckError(clEnqueueNDRangeKernel(Q, kernel_p, 1,
                                      NULL, global_work_size, local_work_size,
                                      0, NULL, &event));
    CheckError(clEnqueueNDRangeKernel(Q, kernel_wall, 1,
                                      NULL, global_work_size_w, local_work_size,
                                      0, NULL, &event));
    for (int i=0; i < nTime; i++)
    {
        if (i % saveFreq == 0)
        {
            saveCheckpoint(to_string(i), output_dir, buf_x, buf_v, buf_vw, buf_r, buf_rw, buf_p, buf_pw, x, v, r, p, Q, Nw);
        }

        CheckError(clEnqueueNDRangeKernel(Q, kernel_dr, 1,
                                          NULL, global_work_size, local_work_size,
                                          0, NULL, &event));
        CheckError(clEnqueueNDRangeKernel(Q, kernel_den, 1,
                                          NULL, global_work_size, local_work_size,
                                          0, NULL, &event));
        CheckError(clEnqueueNDRangeKernel(Q, kernel_p, 1,
                                          NULL, global_work_size, local_work_size,
                                          0, NULL, &event));
        CheckError(clEnqueueNDRangeKernel(Q, kernel_dv, 1,
                                          NULL, global_work_size, local_work_size,
                                          0, NULL, &event));
        CheckError(clEnqueueNDRangeKernel(Q, kernel_vel, 1,
                                          NULL, global_work_size, local_work_size,
                                          0, NULL, &event));
        CheckError(clEnqueueNDRangeKernel(Q, kernel_dx, 1,
                                          NULL, global_work_size, local_work_size,
                                          0, NULL, &event));
        CheckError(clEnqueueNDRangeKernel(Q, kernel_pos, 1,
                                          NULL, global_work_size, local_work_size,
                                          0, NULL, &event));
        CheckError(clEnqueueNDRangeKernel(Q, kernel_wall, 1,
                                          NULL, global_work_size_w, local_work_size,
                                          0, NULL, &event));
    }

    saveCheckpoint("final", output_dir, buf_x, buf_v, buf_vw, buf_r, buf_rw, buf_p, buf_pw, x, v, r, p, Q, Nw);
}
