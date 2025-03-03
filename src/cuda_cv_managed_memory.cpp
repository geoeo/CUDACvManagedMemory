#include "CUDACvManagedMemory/cuda_cv_managed_memory.hpp"
#include <stdexcept>
#include <sstream>
#include "driver_types.h"

using namespace cuda_cv_managed_memory;

CUDAManagedMemory::CUDAManagedMemory(size_t sizeInBytes, uint32_t height, uint32_t width, int type, size_t step):
    size_in_bytes_(sizeInBytes), height_(height), width_(width), type_(type), step_(step) {
    checkCudaError(cudaMallocManaged(&unified_ptr_, sizeInBytes), __FILE__, __LINE__);
}

CUDAManagedMemory::SharedPtr CUDAManagedMemory::fromCvMat(const cv::Mat src){
    // CPU mem is continuous - step should reflect this
    auto size_in_bytes = src.step*src.rows;
    auto shared_ptr = std::shared_ptr<CUDAManagedMemory>(new CUDAManagedMemory(size_in_bytes,src.rows,src.cols, src.type(), src.step),CUDAManagedMemoryDeleter{});
    
    checkCudaError(cudaMemcpy(shared_ptr->getRaw(), &src.data[0], size_in_bytes, cudaMemcpyDefault), __FILE__, __LINE__);
    return shared_ptr;
}

CUDAManagedMemory::SharedPtr CUDAManagedMemory::fromCvGpuMat(const cv::cuda::GpuMat src){
    auto size_in_bytes = src.step*src.rows;
    auto shared_ptr = std::shared_ptr<CUDAManagedMemory>(new CUDAManagedMemory(size_in_bytes,src.rows,src.cols, src.type(), src.step),CUDAManagedMemoryDeleter{});

    checkCudaError(cudaMemcpy(shared_ptr->getRaw(), src.ptr(0), size_in_bytes, cudaMemcpyDeviceToDevice), __FILE__, __LINE__);
    return shared_ptr;
}

CUDAManagedMemory::~CUDAManagedMemory(){
    checkCudaError(cudaFree(unified_ptr_), __FILE__, __LINE__);
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

size_t CUDAManagedMemory::getStep() const {
    return step_;
}

cv::Mat CUDAManagedMemory::getCvMat(cudaStream_t stream){
    // https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/
    // Prefetch output image data to CPU - Seems to be neccessary on Tegra SoC in multithreaded settings
    checkCudaError(cudaStreamAttachMemAsync(stream, unified_ptr_, 0, cudaMemAttachHost), __FILE__, __LINE__);
    checkCudaError(cudaStreamSynchronize(stream), __FILE__, __LINE__);
    checkCudaError( cudaGetLastError(), __FILE__, __LINE__ );

    return cv::Mat(height_, width_, type_, unified_ptr_, step_);
}

cv::cuda::GpuMat CUDAManagedMemory::getCvGpuMat(cudaStream_t stream){
    // https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/
    // Prefetch input image data to GPU - Seems to be neccessary on Tegra SoC in multithreaded settings

    checkCudaError(cudaStreamAttachMemAsync(stream, unified_ptr_, 0, cudaMemAttachGlobal), __FILE__, __LINE__);
    checkCudaError(cudaStreamSynchronize(stream), __FILE__, __LINE__);
    checkCudaError( cudaGetLastError(), __FILE__, __LINE__ );

    return cv::cuda::GpuMat(height_, width_, type_, unified_ptr_, step_);
}

void CUDAManagedMemory::checkCudaError(cudaError_t result, const char *const file, int const line){
    if(result != cudaError_t::cudaSuccess){
        std::stringstream ss;
        ss << "CUDA error at " << file <<":"<<line << " "<< cudaGetErrorString(result) << std::endl;
        throw std::runtime_error(ss.str());
    }
}



