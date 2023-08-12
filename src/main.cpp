#include <cuda.h>

#include <format>
#include <iostream>

void cudaCheckError(const CUresult &result)
{
    if (result == CUDA_SUCCESS) return;

    const char *errorName = nullptr;
    cuGetErrorName(result, &errorName);
    const char *errorString = nullptr;
    cuGetErrorString(result, &errorString);

    throw std::runtime_error(std::format("{}: {}\n", errorName, errorString));
}

template <typename T>
class CUDABuffer
{
   private:
    CUdeviceptr dptr = 0;
    int size = 0;

   public:
    CUDABuffer(int size) : size(size)
    {
        cudaCheckError(cuMemAlloc(&dptr, sizeof(T) * size));
    }

    CUDABuffer(const CUDABuffer &) = delete;

    CUDABuffer(CUDABuffer &&other) : dptr(other.dptr), size(other.size)
    {
        other.dptr = 0;
    }

    ~CUDABuffer() { cudaCheckError(cuMemFree(dptr)); }

    const CUdeviceptr &getDevicePtr() const { return dptr; }

    void copyHtoD(const T *hptr) const
    {
        cudaCheckError(cuMemcpyHtoD(dptr, hptr, sizeof(T) * size));
    }

    void copyDtoH(T *hptr) const
    {
        cudaCheckError(cuMemcpyDtoH(hptr, dptr, sizeof(T) * size));
    }
};

class CUDADevice
{
   private:
    CUdevice device = 0;
    CUcontext context = nullptr;

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

    CUDADevice(const CUDADevice &) = delete;

    CUDADevice(CUDADevice &&other)
        : device(other.device), context(other.context)
    {
        other.device = 0;
        other.context = nullptr;
    }

    ~CUDADevice()
    {
        cuCtxPopCurrent(&context);
        cuCtxDestroy(context);
    }

    void synchronize() const { cudaCheckError(cuCtxSynchronize()); }
};

class CUDAKernel
{
   private:
    CUmodule module = nullptr;
    CUfunction function = nullptr;

   public:
    CUDAKernel(const std::string &filename, const std::string &kernelName)
    {
        cudaCheckError(cuModuleLoad(&module, filename.c_str()));
        cudaCheckError(
            cuModuleGetFunction(&function, module, kernelName.c_str()));
    }

    CUDAKernel(const CUDAKernel &) = delete;

    CUDAKernel(CUDAKernel &&other)
        : module(other.module), function(other.function)
    {
        other.module = nullptr;
        other.function = nullptr;
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