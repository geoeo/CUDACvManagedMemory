#pragma once

#include <cstddef>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>

namespace cuda_cv_managed_memory
{
    /**
     * This struct wrapped a CUDA unified memory ptr. 
     * It should only be used wrapped in a SharedPtr.
     * Therefore, Default and Copy Constructors are disabled.
     */
    struct CUDAManagedMemory {
        using SharedPtr = std::shared_ptr<CUDAManagedMemory>;

        CUDAManagedMemory() = delete;
        CUDAManagedMemory(const CUDAManagedMemory&) = delete;
        CUDAManagedMemory & operator=(const CUDAManagedMemory& other) = delete;
        CUDAManagedMemory(size_t sizeInBytes, uint32_t height, uint32_t width, int type, size_t step);
        ~CUDAManagedMemory();

        static CUDAManagedMemory::SharedPtr fromCvMat(const cv::Mat src);
        static CUDAManagedMemory::SharedPtr fromCvGpuMat(const cv::cuda::GpuMat src);

        /**
         * @brief Returns an unmanaged OpenCV Mat. 
         */
        cv::Mat getCvMat();

        /**
         * @brief Returns an unmanaged OpenCV Cuda Mat.
         */
        cv::cuda::GpuMat getCvGpuMat();

        /**
         * This function exposes the raw ptr. 
         * Make sure the lifetime of the bound datastructures are less than the CUDAManagedMemory struct.
         */
        void * getRaw();

        uint32_t getHeight() const;
        uint32_t getWidth() const;
        int getCvType() const;
        uint32_t sizeInBytes() const;
        size_t getStep() const;
        private:
            // Using unified memory
            void *unified_ptr_;
            uint32_t size_in_bytes_;
            uint32_t height_;
            uint32_t width_;
            int type_;
            size_t step_;
            

    };

    struct CUDAManagedMemoryDeleter
    {
        void operator()(CUDAManagedMemory* p) const { delete p; }
    };

}
