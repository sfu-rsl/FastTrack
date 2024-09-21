#ifndef SEARCH_LOCAL_POINTS_KERNEL_H
#define SEARCH_LOCAL_POINTS_KERNEL_H

#include "KernelInterface.h"
#include "../Frame.h"
#include "../MapPoint.h"
#include "CudaUtils.h"
#include "CudaWrappers/CudaFrame.h"
#include "CudaWrappers/CudaMapPoint.h"
#include "CudaWrappers/CudaKeyPoint.h"
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <sys/stat.h>
#include <sys/types.h>

#define FRAME_GRID_ROWS 48
#define FRAME_GRID_COLS 64
#define KEYPOINTS_PER_CELL 20
#define MAX_NUM_MAPPOINTS 25000

class SearchLocalPointsKernel: public KernelInterface {

    public:
        SearchLocalPointsKernel() { memory_is_initialized = false; };
        void initialize() override;
        void shutdown() override;
        void saveStats(const string &file_path) override;
        void launch() override { cout << "[SearchLocalPointsKernel:] provide input for kernel launch.\n"; };
        void launch(ORB_SLAM3::Frame &F, const vector<ORB_SLAM3::MapPoint*> &vmp, const float th, const bool bFarPoints, const float thFarPoints,
                                int* h_bestLevel, int* h_bestLevel2, int* h_bestDist, int* h_bestDist2, int* h_bestIdx,
                                int* h_bestLevelR, int* h_bestLevelR2, int* h_bestDistR, int* h_bestDistR2, int* h_bestIdxR);
        void setFrame(DATA_WRAPPER::CudaFrame* cudaFrame);
    
    private:
        bool memory_is_initialized;
        DATA_WRAPPER::CudaFrame *d_frame, *h_frame;
        int *h_bestLevel, *h_bestLevel2, *h_bestDist, *h_bestDist2, *h_bestIdx;
        int *d_bestLevel, *d_bestLevel2, *d_bestDist, *d_bestDist2, *d_bestIdx;
        int *h_bestLevelR, *h_bestLevelR2, *h_bestDistR, *h_bestDistR2, *h_bestIdxR;
        int *d_bestLevelR, *d_bestLevelR2, *d_bestDistR, *d_bestDistR2, *d_bestIdxR;
        bool *h_isEmpty, *d_isEmpty;
        bool *h_mbTrackInView, *d_mbTrackInView;
        bool *h_mbTrackInViewR, *d_mbTrackInViewR;
        float *h_mTrackDepth, *d_mTrackDepth;
        int *h_mnTrackScaleLevel, *d_mnTrackScaleLevel;
        float *h_mTrackViewCos, *d_mTrackViewCos;
        float *h_mTrackProjX, *d_mTrackProjX;
        float *h_mTrackProjY, *d_mTrackProjY;
        int *h_mnTrackScaleLevelR, *d_mnTrackScaleLevelR;
        float *h_mTrackViewCosR, *d_mTrackViewCosR;
        float *h_mTrackProjXR, *d_mTrackProjXR;
        float *h_mTrackProjYR, *d_mTrackProjYR;
        uint8_t *h_mDescriptor, *d_mDescriptor;

    public:
        std::vector<std::pair<long unsigned int, double>> data_wrap_time;
        std::vector<std::pair<long unsigned int, double>> mappoints_wrap_time;
        std::vector<std::pair<long unsigned int, double>> frame_wrap_time;
        std::vector<std::pair<long unsigned int, double>> data_transfer_time;
        std::vector<std::pair<long unsigned int, double>> input_data_transfer_time;
        std::vector<std::pair<long unsigned int, double>> frame_data_transfer_time;
        std::vector<std::pair<long unsigned int, double>> mappoints_data_transfer_time;
        std::vector<std::pair<long unsigned int, double>> output_data_transfer_time;
        std::vector<std::pair<long unsigned int, double>> kernel_exec_time;
        std::vector<std::pair<long unsigned int, double>> total_exec_time;
};


#endif 