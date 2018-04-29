#define gamma 1.4
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

float time_step(float cfl, float h0, float c0)
{
    float dt=cfl*h0/c0 ;  
    return dt;
}

float kernel_cubic(float2 xi, float2 xj, float h)
{
    float q = distance(xi, xj) / h;
    float W = 0;
    if (q <= 1.)
        W += 10 / (7 * M_PI * h * h) * (1. - 3 / 2 * q * q * (1 - q / 2));
    if (q > 1. && q < 2.)
        W += 10 / (28 * M_PI * h * h) * pow((2 - q),3);
    return W;
}

float2 kernel_derivative(float2 xi, float2 xj, float h)
{
    float q = distance(xi, xj) / h;
    float dwdq = 0;
    if (q <= 1.)
        dwdq = (9 / 4 * q * q - 3 * q) * 10 / (7 * M_PI * h * h);
    if (q > 1. && q < 2.)
        dwdq = -7.5 * (2 - q) * (2-q) / (7 * M_PI * h * h);

    float2 dW = dwdq * (xi - xj) / q;

    return dW;
}

float art_visc(float2 x_i, float2 x_j, float r_i, float r_j, float2 v_i, float2 v_j, float p_i, float p_j, float h)
{
    float alpha = 1;
    float beta = 1;
    float2 x = (x_i - x_j);
    float neta = 0.1 * h;
    float pia = 0;

    if (dot(x, v_i - v_j) <= 0)
    {
// check
	float ca = (sqrt(1.4 * p_i / r_i) + sqrt(1.4 * p_j / r_j)) / 2;
        float ra = (r_i + r_j) / 2;
        float mu = h * dot(v_i - v_j, x) / (pow(length(x), 2) + neta*neta);
        pia = (-alpha * ca * mu + beta * mu * mu) / ra;
    }
    return pia;
}


__kernel void UPDATE_POS(__global float2* x, __global float2* v, __global float* r, float m, float h, int N, float dt)
{

	const int i = get_global_id(0);
    float2 tmp = 0;
    float e_con = 1; //CHANGE THIS

    for(int j = 0; j < N; j++)
    {
        float W = kernel_cubic(x[i], x[j], h);
        tmp += e_con * (m * 2 / (r[j] + r[i])) * (v[j] - v[i]) * W;
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    v[i] = tmp;
    x[i] = v[i] * dt;
}

// N = number of fluid particles. Nw = wall particles. Launch one kernel per fluid particle
__kernel void SUMDEN(__global float2* x, __global float2* xw, __global float* r, float m, float h, int N, int Nw)
//__kernel void SUMDEN(__global float2* x, __global float* r, float m, float h, int N)
{
	const int i = get_global_id(0);
    r[i] = 0;

    for (int j=0; j<N; j++)
        r[i] += m * kernel_cubic(x[i], x[j], h);
    for (int j=0; j<Nw; j++)
        r[i] += m * kernel_cubic(x[i], xw[j], h);
    
}

// Launch one kernel per FLUID particle
__kernel void INCOMP_P(__global float* r,__global float* p, float c0, float rho0)
{
	const int i = get_global_id(0);
    float B = rho0 * c0 * c0 / gamma;
    double tmp = pow((double)r[i] / rho0, (double)gamma) - 1;
    p[i] = B * (tmp) ;
    
}


// Launch one kernel per FLUID particle
__kernel void UPDATE_VEL(__global float2* x, __global float2* xw, __global float* p, __global float* pw, __global float2* v, __global float2* vw,__global float* r, __global float* rw, float m, int N, int Nw, float dt, float h)
{
	const int i = get_global_id(0);
    float2 dv = 0;
	
    for (int j=0; j<N; j++)
    {
        if (distance(x[i], x[j]) < 2*h)
        {
            float av = art_visc(x[i], x[j], r[i], r[j], v[i], v[j], p[i], p[j], h);
            float2 dW = kernel_derivative(x[i], x[j], h);
        
            float calc = (p[j] / r[j] / r[j] + p[i] / r[i] / r[i] + av);

            dv += - m * calc * dW;
        }
    }

    for (int j=0; j<Nw; j++)
    {
        if (distance(x[i], xw[j]) < 2*h)
        {
            float av = art_visc(x[i], xw[j], r[i], rw[j], v[i], vw[j], p[i], pw[j], h);
            float2 dW = kernel_derivative(x[i], xw[j], h);
        
            float calc = (pw[j] / rw[j] / rw[j] + p[i] / r[i] / r[i] + av);

            dv += - m * calc * dW;
        }
    }
    v[i] += dv * dt;
}

// Launch one kernel per wall particle
__kernel void WALL(__global float2* x, __global float2* v, __global float2* vw, __global float* p, __global float* pw, __global float* rw, float h, float rho0, float c0, int N)
{
	const int i = get_global_id(0);
    float2 num_v_w = 0;
    float num_p_w =0;
    float den_w = 0;
	
    for (int j=0; j<N ; j++)
    {
    	num_v_w += v[j]*kernel_cubic(x[i], x[j], h);
        num_p_w += p[j]*kernel_cubic(x[i], x[j], h);
        den_w += kernel_cubic(x[i], x[j], h);
    }

    vw[i] = - num_v_w/den_w;
    pw[i] = num_p_w/den_w;
	
    float B = rho0 * c0 * c0 / gamma;
    double tmp =   pw[i]/B + 1; 
    double tmp1 = 1/gamma;
    double tmp2 = pow((double) tmp, (double) tmp1);
    rw[i] = rho0 * (tmp2) ;
}


__kernel void MIRROR(__global float2* x, __global float2* geom, __global float2* u, int N, int geom_numpts)
{

}
__kernel void REDUCE(__global float *blk_sum, __global float *ret, 
                     __local float *y, int numItems)
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
