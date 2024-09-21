#ifndef KERNEL_CONTROLLER_H
#define KERNEL_CONTROLLER_H

#include "SearchLocalPointsKernel.h"
#include "PoseEstimationKernel.h"
#include "StereoMatchKernel.h"
#include "CudaWrappers/CudaFrame.h"
#include "CudaUtils.h"
#include "../Stats.h"
#include <memory> 
using namespace std; 

class KernelController{
public:
    static void setCUDADevice(int deviceID);
    
    static void setGPURunMode(bool orbExtractionStatus, bool stereoMatchStatus, bool searchLocalPointsStatus, bool poseEstimationStatus, bool poseOptimizationStatus);

    static bool orbExtractionKernelRunStatus;
    static bool stereoMatchKernelRunStatus;
    static bool searchLocalPointsKernelRunStatus;
    static bool poseEstimationKernelRunStatus;
    static bool poseOptimizationRunStatus;

    static void initializeKernels();
    
    static void shutdownKernels();
    
    static void saveKernelsStats(const std::string &file_path);
    
    static void launchStereoMatchKernel(std::vector<std::vector<int>> &vRowIndices, uchar* d_imagePyramidL, uchar* d_imagePyramidR, 
                                        std::vector<cv::Mat> &mvImagePyramid, std::vector<cv::Mat> &mvImagePyramidRight,
                                        std::vector<cv::KeyPoint> &mvKeys, std::vector<cv::KeyPoint> &mvKeysRight, 
                                        cv::Mat mDescriptors, cv::Mat mDescriptorsRight, const float minD, const float maxD, const int thOrbDist, 
                                        const float mbf, const bool mvImagePyramidOnGpu, 
                                        std::vector<std::pair<int, int>> &vDistIdx, std::vector<float> &mvuRight, std::vector<float> &mvDepth);
    
    static void launchFisheyeStereoMatchKernel(const int N, const int Nr, cv::Mat mDescriptors, cv::Mat mDescriptorsRight, int* matches);

    static void launchSearchLocalPointsKernel(ORB_SLAM3::Frame &F, const vector<ORB_SLAM3::MapPoint*> &vmp, const float th, const bool bFarPoints, const float thFarPoints,
                                            int* h_bestLevel, int* h_bestLevel2, int* h_bestDist, int* h_bestDist2, int* h_bestIdx,
                                            int* h_bestLevelR, int* h_bestLevelR2, int* h_bestDistR, int* h_bestDistR2, int* h_bestIdxR);

    static void launchPoseEstimationKernel(ORB_SLAM3::Frame &CurrentFrame, const ORB_SLAM3::Frame &LastFrame, 
                                            const float th, const bool bForward, const bool bBackward, Eigen::Matrix4f transform_matrix,
                                            int* h_bestDist, int* h_bestIdx2, int* h_bestDistR, int* h_bestIdxR2);

private:
    static bool memory_is_initialized;
    static bool stereoMatchDataHasMovedForward;
    static DATA_WRAPPER::CudaFrame *cudaFramePtr, *cudaLastFramePtr;
    static std::unique_ptr<StereoMatchKernel> mpStereoMatchKernel;
    static std::unique_ptr<SearchLocalPointsKernel> mpSearchLocalPointsKernel;
    static std::unique_ptr<PoseEstimationKernel> mpPoseEstimationKernel;
};

#endif