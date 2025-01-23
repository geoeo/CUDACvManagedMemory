#include "CUDACvManagedMemory/cuda_cv_managed_memory.hpp"
#include <stdexcept>
#include "cuda_runtime.h"
#include "driver_types.h"

using namespace cuda_cv_managed_memory;

CUDAManagedMemory::CUDAManagedMemory(size_t sizeInBytes, uint32_t height, uint32_t width, int type, size_t step):
    size_in_bytes_(sizeInBytes), height_(height), width_(width), type_(type), step_(step) {
    const auto cuda_res = cudaMallocManaged(&unified_ptr_, sizeInBytes);
    if(cuda_res != cudaError_t::cudaSuccess)
        throw std::runtime_error("CUDAManagedMemory - Runtime Cuda Error: " + std::to_string(cuda_res));
}

CUDAManagedMemory::SharedPtr CUDAManagedMemory::fromCvMat(const cv::Mat src){
    // CPU mem is continuous
    auto size_in_bytes = src.cols*src.rows*src.channels();
    auto shared_ptr = std::shared_ptr<cuda_managed_memory::CUDAManagedMemory>(new CUDAManagedMemory(size_in_bytes,src.rows,src.cols, src.type(), src.step),CUDAManagedMemoryDeleter{});
    if(cudaMemcpy(shared_ptr->getRaw(), &src.data[0], size_in_bytes, cudaMemcpyDefault) != cudaError_t::cudaSuccess)
        throw std::runtime_error("CUDAManagedMemory - Failed to copy memory to CUDA unified");
    return shared_ptr;
}

CUDAManagedMemory::SharedPtr CUDAManagedMemory::fromCvGpuMat(const cv::cuda::GpuMat src){
    auto size_in_bytes = src.step*src.rows;
    auto shared_ptr = std::shared_ptr<cuda_managed_memory::CUDAManagedMemory>(new CUDAManagedMemory(size_in_bytes,src.rows,src.cols, src.type(), src.step),CUDAManagedMemoryDeleter{});

    if(cudaMemcpy(shared_ptr->getRaw(), src.ptr(0), size_in_bytes, cudaMemcpyDeviceToDevice) != cudaError_t::cudaSuccess)
        throw std::runtime_error("CUDAManagedMemory - Failed to copy memory to CUDA unified");
    return shared_ptr;
}

CUDAManagedMemory::~CUDAManagedMemory(){
    cudaFree(unified_ptr_);
}

void* CUDAManagedMemory::getRaw() {
    return unified_ptr_;
}

uint32_t CUDAManagedMemory::getHeight() const {
    return height_;
}

uint32_t CUDAManagedMemory::getWidth() const {
    return width_;
}

uint32_t CUDAManagedMemory::sizeInBytes() const {
    return size_in_bytes_;
}

int CUDAManagedMemory::getCvType() const {
    return type_;
}

cv::Mat CUDAManagedMemory::getCvMat(){
    // https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/
    // Prefetch output image data to CPU - Seems to be neccessary on Tegra SoC in multithreaded settings
    cudaStreamAttachMemAsync(NULL, unified_ptr_, 0, cudaMemAttachHost);
    cudaStreamSynchronize(NULL);
    return cv::Mat(height_, width_, type_, unified_ptr_, step_);
}

cv::cuda::GpuMat CUDAManagedMemory::getCvGpuMat(){
    // https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/
    // Prefetch input image data to GPU - Seems to be neccessary on Tegra SoC in multithreaded settings
    cudaStreamAttachMemAsync(NULL, unified_ptr_, 0, cudaMemAttachGlobal);
    cudaStreamSynchronize(NULL);
    return cv::cuda::GpuMat(height_, width_, type_, unified_ptr_, step_);
}



