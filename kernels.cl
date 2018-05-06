#define gamma 1.4
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

double time_step(double cfl, double h0, double c0)
{
    double dt=cfl*h0/c0 ;  
    return dt;
}

double kernel_cubic(double2 xi, double2 xj, double h)
{
    double q = distance(xi, xj) / h;
    double W = 0;
    if (q <= 1.&& q >= 0)
        W += 10 / (7 * M_PI * h * h) * (1. - 3 / 2 * q * q * (1 - q / 2));
    if (q > 1. && q < 2.)
        W += 10 / (28 * M_PI * h * h) * pow((2 - q),3);
    return W;
}

double2 kernel_derivative(double2 xi, double2 xj, double h)
{
    double q = distance(xi, xj) / h;
    double dwdq = 0;
    if (q <= 1.)
        dwdq = (9 / 4 * q - 3) * 10 / (7 * M_PI * h * h);
    if (q > 1. && q < 2.)
        dwdq = -7.5 * (2 - q) * (2-q) / (7 * M_PI * q * h * h);

    double2 dW = dwdq * (xi - xj) / h / h;

    return dW;
}

double art_visc(double2 x_i, double2 x_j, double r_i, double r_j, double2 v_i, double2 v_j, double p_i, double p_j, double h)
{
    double alpha = 1;
    double beta = 1;
    double2 x = (x_i - x_j);
    double neta = 0.01 * h;
    double pia = 0;

    if (dot(x, v_i - v_j) <= 0)
    {
        double ca = (sqrt(fabs(1.4 * p_i / r_i)) + sqrt(fabs(1.4 * p_j / r_j))) / 2;
        double ra = (r_i + r_j) / 2;
        double mu = h * dot(v_i - v_j, x) / (pow(length(x), 2) + neta*neta);
        pia = (-alpha * ca * mu + beta * mu * mu) / ra;
    }
    return pia;
}


// Launch one kernel per FLUID particle
__kernel void DELTA_X(__global double2* x, __global double2* xw, __global double2* dx, __global double2* v, __global double2* vw, __global double* r, __global double* rw, double m, double h, int N, int Nw, double dt)
{

	const int i = get_global_id(0);

    if (i < N){
        dx[i] = 0;
        double2 tmp = 0;
        double e_con = 0.5;

        for(int j = 0; j < N; j++)
        {
            double W = kernel_cubic(x[i], x[j], h);
            tmp += e_con * (m * 2 / (r[j] + r[i])) * (v[j] - v[i]) * W;
        }

        for(int j = 0; j < Nw; j++)
        {
            double W = kernel_cubic(x[i], xw[j], h);
            tmp += e_con * (m * 2 / (rw[j] + r[i])) * (vw[j] - v[i]) * W;
        }

        dx[i] += (tmp + v[i]) * dt;
    }
}

// Launch one kernel per FLUID particle
__kernel void UPDATE_POS(__global double2* x, __global double2* dx, int N)
{
    const int i = get_global_id(0);
    if (i < N){
        x[i] += dx[i];
    }
}

// Launch one kernel per FLUID particle
__kernel void SUMDEN(__global double2* x, __global double2* xw, __global double* r, double m, double h, int N, int Nw)
{
	const int i = get_global_id(0);

    if (i < N){
        r[i] = 0;

        for (int j=0; j<N; j++)
            r[i] += m * kernel_cubic(x[i], x[j], h);
        for (int j=0; j<Nw; j++)
            r[i] += m * kernel_cubic(x[i], xw[j], h);
    }
}

// Launch one kernel per FLUID particle
__kernel void DELTA_DEN(__global double2* x, __global double2* xw, __global double2* v, __global double2* vw, __global double* r, __global double* rw, __global double* dr, double m, double h, double dt, int N, int Nw)
{
	const int i = get_global_id(0);
    if (i < N){
        dr[i] = 0;
        double tmp = 0;
        for (int j=0; j<N; j++)
        {
            tmp += 1 / r[j] * dot(v[i] - v[j], kernel_derivative(x[i], x[j], h));
        }
        for (int j=0; j<Nw; j++)
        {
            tmp += 1 / rw[j] * dot(v[i] - vw[j], kernel_derivative(x[i], xw[j], h));
        }
        dr[i] = tmp * m * r[i] * dt;
    }
}

// Launch one kernel per FLUID particle
__kernel void UPDATE_DEN(__global double* r, global double* dr, int N)
{
    const int i = get_global_id(0);
    if (i < N){
        r[i] += dr[i];
    }
}

// Launch one kernel per FLUID particle
__kernel void INCOMP_P(__global double* r,__global double* p, double c0, double rho0, int N)
{
	const int i = get_global_id(0);
    double chi = 0.05;

    if (i < N){
        double B = rho0 * c0 * c0 / gamma;
        double tmp = pow(r[i] / rho0, gamma) - 1;
        p[i] = B * (tmp + chi);
    }
    
}


// Launch one kernel per FLUID particle
__kernel void DELTA_V(__global double2* x, __global double2* xw, __global double* p, __global double* pw, __global double2* v, __global double2* vw, __global double2* dv, __global double* r, __global double* rw, double m, int N, int Nw, double dt, double h)
{
	const int i = get_global_id(0);

    if (i < N){
        dv[i] = 0;
        
        for (int j=0; j<N; j++)
        {
            if (distance(x[i], x[j]) < 2*h)
            {
                double av = art_visc(x[i], x[j], r[i], r[j], v[i], v[j], p[i], p[j], h);
                double2 dW = kernel_derivative(x[i], x[j], h);
            
                double calc = (p[j] / r[j] / r[j] + p[i] / r[i] / r[i] + av);

                dv[i] += - m * calc * dW * dt;
            }
        }

        for (int j=0; j<Nw; j++)
        {
            if (distance(x[i], xw[j]) < 2*h)
            {
                double av = art_visc(x[i], xw[j], r[i], rw[j], v[i], vw[j], p[i], pw[j], h);
                double2 dW = kernel_derivative(x[i], xw[j], h);
            
                double calc = (pw[j] / rw[j] / rw[j] + p[i] / r[i] / r[i] + av);

                dv[i] += - m * calc * dW * dt;
            }
        }
    }
}


__kernel void UPDATE_VEL(__global double2* v, __global double2* dv, int N)
{
    const int i = get_global_id(0);

    if (i < N){
        v[i] += dv[i];
    }
}


// Launch one kernel per wall particle
__kernel void WALL(__global double2* x, __global double2* xw, __global double2* v, __global double2* vw, __global double* p, __global double* pw, __global double* rw, double h, double rho0, double c0, int N, int Nw)
{
	const int i = get_global_id(0);
    double chi = 0.05;

    if (i < Nw){
        double2 num_v_w = 0;
        double num_p_w =0;
        double den_w = 0;
        
        for (int j=0; j<N ; j++)
        {
            double kernel_output = kernel_cubic(xw[i], x[j], h);
            num_v_w += v[j] * kernel_output;
            num_p_w += p[j] * kernel_output;
            den_w += kernel_output;
        }

        if (fabs(den_w) > 1e-16){
            vw[i] =  -(num_v_w/den_w);
            pw[i] = num_p_w/den_w;
        }

        else{
            vw[i] = 0;
            pw[i] = 0;
        }
        
        double B = rho0 * c0 * c0 / gamma;
        double tmp = pw[i]/B + 1 - chi; 
        double tmp1 = 1/gamma;
        double tmp2 =  pow((double) tmp, (double) tmp1);
        rw[i] = rho0 * (tmp2) ;

        if ( rw[i] < rho0 ) rw[i] = rho0;
    }
}

__kernel void REDUCE(__global double *blk_sum, __global double *ret, 
                     __local double *y, int numItems)
{
    const int tid = get_local_id(0);
    int blk_size = get_local_size(0);    

    int i = tid;
    y[tid] = 0.0;
    while (i < numItems) {
        y[tid] += blk_sum[i];
        i += blk_size;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = blk_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            y[tid] += y[tid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (tid == 0) ret[0] = y[0];

}
