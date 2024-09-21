#ifndef CUDA_FRAME_H
#define CUDA_FRAME_H

#include <vector>
#include "Frame.h"
#include "../CudaUtils.h"
#include "CudaMapPoint.h"
#include "CudaKeyPoint.h"
#include <opencv2/opencv.hpp>
#include <memory>
#include <Eigen/Dense>
#include <sophus/se3.hpp>
#include "../StereoMatchKernel.h"

namespace DATA_WRAPPER {

#define FRAME_GRID_ROWS 48
#define FRAME_GRID_COLS 64
#define KEYPOINTS_PER_CELL 20

// CudaFrameMemorySpace
class CudaFrame {
    private:
        void initializeMemory();
    
    private:
        bool mvKeysIsOnGpu, mvKeysRightIsOnGpu, mDescriptorsIsOnGpu;

    public:
        CudaFrame();
        void setMemory(const ORB_SLAM3::Frame &F);
        void setMvKeys(CudaKeyPoint* const &_mvKeys);
        void setMvKeysRight(CudaKeyPoint* const &_mvKeysRight);
        void setMDescriptors(uint8_t* const &_mDescriptors);
        const uint8_t* getMDescriptors() const { return mDescriptors; };
        void freeMemory();
    
    public:
        long unsigned int mnId;
        int Nleft;
        int N;
        float mnMinX;
        float mnMinY;
        float mnMaxX;
        float mnMaxY;
        float mfGridElementWidthInv;
        float mfGridElementHeightInv;
        float mbf;

        size_t mvpMapPoints_size;
        CudaMapPoint* mvpMapPoints;

        size_t mvuRight_size;
        float* mvuRight;

        int mDescriptor_rows;
        const uint8_t* mDescriptors;

        size_t mvKeys_size, mvKeysRight_size, mvKeysUn_size;
        const CudaKeyPoint *mvKeys, *mvKeysRight;
        CudaKeyPoint *mvKeysUn;

        size_t flatMGrid_size[FRAME_GRID_COLS * FRAME_GRID_ROWS];
        std::size_t flatMGrid[FRAME_GRID_COLS * FRAME_GRID_ROWS * KEYPOINTS_PER_CELL];        
        
        size_t flatMGridRight_size[FRAME_GRID_COLS * FRAME_GRID_ROWS];
        std::size_t flatMGridRight[FRAME_GRID_COLS * FRAME_GRID_ROWS * KEYPOINTS_PER_CELL];

        size_t mvbOutlier_size;
        uint8_t* mvbOutlier;
        
        float mpCamera_mvParameters[8];
        
        Eigen::Matrix4f mTrl;

    };
}

#endif // CUDA_FRAME_H