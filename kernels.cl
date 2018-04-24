#DEFINE gamma 1.4

// test

double kernel_cubic(float2 xi, float2 xj, double h)
{
    double q = distance(xi, xj);
    double W = 0;
    if (q <= 1.)
        W += 2. / (3 * h) * (1. - 3 / 2 * q * q * (1 - q / 2));
    if (q > 1. && q < 2.)
        W += 2. / (12 * h) * pow((2 - q),3);
    return W
}

double kernel_derivative(float2 xi, float2 xj, double h)
{
    double q = distance(xi, xj);
    double dW = 0;
    if (q <= 1.)
        dW = (3 * q * q - 4 * q) / (2 * h);
    if (q > 1. && q < 2.)
        dW = - (2 - q) * (2-q) / (2 * h);

    double der = 0.;
    if (xi > xj)
        der = 1 / h;
    if (xi < xj)
        der = -1 / h;
    
    return dW * der;
}

double art_visc(float2 x_i, float2 x_j, double r_i, double r_j, float2 v_i, float2 v_j, double p_i, double p_j, double h)
{
    float2 x = (x_i - x_j);
    double neta = 0.1 * h;
    double pia = 0;

    if (x * (v_i - v_j) <= 0)
        double ca = (pow(abs(gamma * p_i / r_i), 0.5) +
              pow(abs(gamma * p_j / r_j), 0.5)) / 2;
        double ra = (r_i + r_j) / 2;
        double mu = h * (v_i - v_j) * x / (np.abs(x)**2 + neta**2);
        pia = (-alpha * ca * mu + beta * mu**2) / ra;

    return pia;
}


__kernel void VEL(__global float2* x, __global float2* v, __global double* r, double m, __global float2* u, double h, int N)
{

	const int i = get_global_id(0);

    for(int j = 0; j < N; j++)
    {
        double W = kernel_cubic(x[i], x[j], h);
        v[i] += e_con * (m * 2 / (r[j] + r[i])) * (u[j] - u[i]) * W;
    }
}

__kernel void SUMDEN(__global float2* x, __global double* r, double m, int N)
{
	const int i = get_global_id(0);
    r[i] = 0;

    for (int j=0; j<N; j++)
        r[i] += m * kernel_cubic(x[i], x[j], h);
    
}

__kernel void UPDATE_VEL(__global float2* x, __global float2* v, __global double* r, double m, int N)
{
	const int i = get_global_id(0);
    double dv = 0;
    double p_i = (gamma - 1) * r[i] * e[i];

    for (int j=0; j<N; j++)
    {
        double p_j = (gamma - 1) * r[j] * e[j];

        double av = art_visc(x[i], x[j], r[i], r[j], v[i], v[j], p_i, p_j, h);
        double dW = kernel_derivative(x[i], x[j]);
        
        double calc = (p_j / r_j**2 + p_i / r_i**2 + av);
        double calc1 = (p_i / r_i**2 + av);

        dv += - m * calc * dW;
        de += m / 2 * calc1 * (v[i] - v[j]) * dW;
    }

    v[i] += dv * dt;
    //e[i] += de * dt;
}

__kernel void MIRROR(__global float2* x, __global float2* geom, __global float2* u, int N, int geom_numpts)
{

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
