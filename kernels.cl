double kernel_cubic(float2 xi, float2 xj)
{
    double q = distance(xi, xj);
    double W = 0;
    if (q <= 1.)
        W += 2. / (3 * h) * (1. - 3 / 2 * q * q * (1 - q / 2));
    if (q > 1. && q < 2.)
        W += 2. / (12 * h) * pow((2 - q),3);
    return W
}

double kernel_derivative(float2 xi, float2 xj)
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


__kernel void VEL(__global float2* x, __global float2* v, __global double* r, double m, __global float2* u, double h, int N)
{

	const int i = get_global_id(0);

    for(int j = 0; j < N; j++)
    {
        q = distance(x[i], x[j]);
        double W = kernel_cubic(q);
        v[i] += e_con * (m * 2 / (r[j] + r[i])) * (u[j] - u[i]) * W;
    }
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
