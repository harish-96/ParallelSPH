__kernel void CUBIC(__global float2* x, __global double *W, int N)
{

	const int i = get_global_id(0);
    const int tid = get_local_id(0);
    const int numItems = get_local_size(0);


    for(int j = 0; j < N; j++)
    {
        q = distance(x[i], x[j]);
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = numItems / 2; s > 0; s >>= 1) {
        if (tid < s && tid + s < numItems) {
            y[tid] += y[tid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        }

    if (tid == 0) blk_sum[get_group_id(0)] = y[0];
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
