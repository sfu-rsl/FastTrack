#ifndef POSE_ESTIMATION_KERNEL_H
#define POSE_ESTIMATION_KERNEL_H

#include "KernelInterface.h"
#include "CudaWrappers/CudaFrame.h"
#include "../Frame.h"
#include "CudaUtils.h"
#include <map>
#include <fstream>
#include <iostream>
#include <sys/stat.h>
#include <sys/types.h>

class PoseEstimationKernel: public KernelInterface {

    public:
        PoseEstimationKernel(){memory_is_initialized=false;};
        void initialize() override;
        void shutdown() override;
        void saveStats(const string &file_path) override;
        void launch() override { cout << "[PoseEstimationKernel:] provide input for kernel launch.\n"; };
        void launch(ORB_SLAM3::Frame &CurrentFrame, const ORB_SLAM3::Frame &LastFrame,
                    const float th, const bool bForward, const bool bBackward, Eigen::Matrix4f transform_matrix,
                    int* h_bestDist, int* h_bestIdx2, int* h_bestDistR, int* h_bestIdxR2);
        void setCurrentFrame(DATA_WRAPPER::CudaFrame* cudaFrame);
        void setLastFrame(DATA_WRAPPER::CudaFrame* cudaFrame);
        DATA_WRAPPER::CudaFrame* getLastFrame() { return d_lastFrame; };
        DATA_WRAPPER::CudaFrame* getCurrentFrame() { return d_currentFrame; };
    
    private:
        bool memory_is_initialized;
        DATA_WRAPPER::CudaFrame *d_currentFrame, *h_currentFrame;
        DATA_WRAPPER::CudaFrame *d_lastFrame, *h_lastFrame;
        int *d_bestDist, *d_bestIdx2, *d_bestDistR, *d_bestIdxR2;

    public:
        std::vector<std::pair<long unsigned int, double>> total_exec_time;
        std::vector<std::pair<long unsigned int, double>> data_wrap_time;
        std::vector<std::pair<long unsigned int, double>> data_transfer_time;
        std::vector<std::pair<long unsigned int, double>> kernel_exec_time;
        std::vector<std::pair<long unsigned int, double>> input_data_transfer_time;
        std::vector<std::pair<long unsigned int, double>> output_data_transfer_time;
};

#endif