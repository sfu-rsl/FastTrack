#ifndef STEREOMATCHKERNEL_H
#define STEREOMATCHKERNEL_H

#include "ORBextractor.h"
#include "CudaUtils.h"
#include "CudaWrappers/CudaKeyPoint.h"
#include "KernelInterface.h"
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <sys/stat.h>
#include <sys/types.h>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#else
#include <cuda_runtime_api.h>
#endif

#define MAX_FEATURES_IN_ROW_SLIDING_WINDOW 200
#define SLIDING_WINDOW_SIZE_W 4
#define SLIDING_WINDOW_SEARCH_SIZE_L 5

class StereoMatchKernel : public KernelInterface {
public:
    StereoMatchKernel() { memory_is_initialized = false;
                          mvImagePyramidOnGpu = false; 
                          frameCounter = 0; };
    void initialize() override;
    void shutdown() override;
    void saveStats(const std::string &file_path) override;
    void launch() override { std::cout << "[StereoMatchKernel:] provide input for kernel launch.\n"; };
    void launch(std::vector<std::vector<int>> &vRowIndices, uchar* d_imagePyramidL, uchar* d_imagePyramidR, 
                std::vector<cv::Mat> &mvImagePyramid, std::vector<cv::Mat> &mvImagePyramidRight,
                std::vector<cv::KeyPoint> &mvKeys, std::vector<cv::KeyPoint> &mvKeysRight, cv::Mat mDescriptors, cv::Mat mDescriptorsRight, 
                const float minD, const float maxD, const int thOrbDist, const float mbf, const bool mvImagePyramidOnGpu,
                std::vector<std::pair<int, int>> &vDistIdx, std::vector<float> &mvuRight, std::vector<float> &mvDepth);
    void launch(const int N, const int Nr, cv::Mat mDescriptors, cv::Mat mDescriptorsRight, int* matches);

private:
    std::vector<std::pair<int, int>> convertToVectorOfPairs(int* X, int N);
    void flattenVRowIndices(const std::vector<std::vector<int>>& input, int* flat);
    void flattenPyramid(std::vector<cv::Mat>& mvImagePyramid, int origImageSize, uchar* flat);
    void copyGPUKeypoints(const std::vector<cv::KeyPoint> keypoints, DATA_WRAPPER::CudaKeyPoint* out);

private:
    std::vector<std::pair<long unsigned int, double>> data_wrap_time;
    std::vector<std::pair<long unsigned int, double>> input_data_wrap_time;
    std::vector<std::pair<long unsigned int, double>> input_data_transfer_time;
    std::vector<std::pair<long unsigned int, double>> kernel1_exec_time;
    std::vector<std::pair<long unsigned int, double>> kernel2_exec_time;
    std::vector<std::pair<long unsigned int, double>> kernel_exec_time;
    std::vector<std::pair<long unsigned int, double>> output_data_transfer_time;
    std::vector<std::pair<long unsigned int, double>> data_transfer_time;
    std::vector<std::pair<long unsigned int, double>> output_data_wrap_time;
    std::vector<std::pair<long unsigned int, double>> total_exec_time;

    long unsigned int frameCounter;

public:
    DATA_WRAPPER::CudaKeyPoint* get_d_gpuKeypointsL() { return d_gpuKeypointsL; };
    DATA_WRAPPER::CudaKeyPoint* get_d_gpuKeypointsR() { return d_gpuKeypointsR; };
    uchar* get_d_descriptorsL() { return d_descriptorsL; };
    uchar* get_d_descriptorsR() { return d_descriptorsR; };
    // For fisheye
    uchar* get_d_descriptorsAll() { return d_descriptorsAll; };

private:
    bool memory_is_initialized;
    int *d_rowIndices;
    uchar *d_imagePyramidL, *d_imagePyramidR;
    uchar *d_imagePyramidLCopied, *d_imagePyramidRCopied;
    DATA_WRAPPER::CudaKeyPoint *d_gpuKeypointsL, *d_gpuKeypointsR;
    uchar *d_descriptorsL, *d_descriptorsR, *d_descriptorsAll;
    int *d_vDistIdx;
    float *d_mvuRight, *d_mvDepth;
    int *d_bestIdxR;
    int *d_matches;
    bool mvImagePyramidOnGpu;
};

#endif