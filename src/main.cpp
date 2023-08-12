#include <cuda.h>

#include <format>
#include <iostream>

void cudaCheckError(const CUresult &result)
{
    const char *pStr;
    if (cuGetErrorName(result, &pStr) != CUDA_ERROR_INVALID_VALUE) {
        std::string str(pStr);
        if (str != "CUDA_SUCCESS") { throw std::runtime_error(pStr); }
    }
}

template <typename T>
class CUDABuffer
{
   private:
    CUdeviceptr dptr;
    int size;

   public:
    CUDABuffer(int size) : size(size)
    {
        cudaCheckError(cuMemAlloc(&dptr, sizeof(T) * size));
    }

    ~CUDABuffer() { cudaCheckError(cuMemFree(dptr)); }

    const CUdeviceptr &getDevicePtr() const { return dptr; }

    void copyHtoD(const T *hptr)
    {
        cudaCheckError(cuMemcpyHtoD(dptr, hptr, sizeof(T) * size));
    }

    void copyDtoH(T *hptr)
    {
        cudaCheckError(cuMemcpyDtoH(hptr, dptr, sizeof(T) * size));
    }
};

class CUDADevice
{
   private:
    CUdevice device;
    CUcontext context;

   public:
    CUDADevice(CUdevice device) : device(device)
    {
        // check device availability
        int nDevices = 0;
        cudaCheckError(cuDeviceGetCount(&nDevices));
        if (device >= nDevices) {
            throw std::runtime_error(
                std::format("device {} is not available\n", device));
        }

        cudaCheckError(cuCtxCreate(&context, CU_CTX_SCHED_AUTO, device));
        cuCtxPushCurrent(context);
    }

    ~CUDADevice()
    {
        cuCtxPopCurrent(&context);
        cuCtxDestroy(context);
    }
};

class CUDAKernel
{
   private:
    CUmodule module;
    CUfunction function;

   public:
    CUDAKernel(const std::string &filename, const std::string &kernelName)
    {
        cudaCheckError(cuModuleLoad(&module, filename.c_str()));
        cudaCheckError(
            cuModuleGetFunction(&function, module, kernelName.c_str()));
    }

    ~CUDAKernel() { cudaCheckError(cuModuleUnload(module)); }

    void launch(const int gridX, const int gridY, const int gridZ,
                const int blockX, const int blockY, const int blockZ,
                const void *args[]) const
    {
        cudaCheckError(cuLaunchKernel(function, gridX, gridY, gridZ, blockX,
                                      blockY, blockZ, 0, nullptr,
                                      const_cast<void **>(args), nullptr));
    }
};

int main()
{
    // init CUDA
    cudaCheckError(cuInit(0));

    // init CUDA context
    CUDADevice device(0);

    constexpr int N = 10;

    float a[N], b[N], c[N];
    for (int i = 0; i < N; ++i) {
        a[i] = i;
        b[i] = i;
    }

    CUDABuffer<float> ad(N);
    ad.copyHtoD(a);
    CUDABuffer<float> bd(N);
    bd.copyHtoD(b);
    CUDABuffer<float> cd(N);

    CUDAKernel kernel("CMakeFiles/kernel.dir/src/kernel.ptx", "addKernel");
    const void *args[] = {&ad.getDevicePtr(), &bd.getDevicePtr(),
                          &cd.getDevicePtr(), &N};
    kernel.launch(1, 1, 1, N, 1, 1, args);

    cd.copyDtoH(c);
    for (int i = 0; i < N; ++i) { std::cout << c[i] << std::endl; }
}