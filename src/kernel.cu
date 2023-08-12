extern "C" __global__ void addKernel(const float *a, const float *b, float *c,
                                     int N)
{
    int i = threadIdx.x;
    if (i >= N) return;

    c[i] = a[i] + b[i];
}