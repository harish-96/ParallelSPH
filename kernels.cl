__kernel void VEL(__global float2* x, __global double* v, __global double* r, __global double* m, __global double* u, double h, int N)
{

	const int i = get_global_id(0);

    for(int j = 0; j < N; j++)
    {
        q = distance(x[i], x[j]);
        double tmp = e_con * (m[j] * 2 / (r[j] + r[i])) * (u[j] - u[i]);
        if (q <= 1)
            v[i] += tmp * 2. / (3 * h) * (1. - 3 / 2 * q * q * (1 - q / 2));
        if (q > 1 && q < 2)
            v[i] += tmp * 2. / (12 * h) * pow((2 - q), 3);
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
