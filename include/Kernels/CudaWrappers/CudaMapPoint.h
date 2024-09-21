#ifndef CUDA_MAPPOINT_H
#define CUDA_MAPPOINT_H

#include "MapPoint.h"
#include <opencv2/opencv.hpp>
#include <Eigen/Core>

namespace DATA_WRAPPER {

class CudaMapPoint {
    public:
        CudaMapPoint();
        CudaMapPoint(ORB_SLAM3::MapPoint* mp);
    
    public:
        // For creating empty mapPoints instead of using null ptr
        bool isEmpty;

    // For searchByProjection in TrackLocalMap
    public:
        long unsigned int mnId;
        bool mbBad;
        bool mbTrackInView;
        bool mbTrackInViewR;
        float mTrackDepth;
        int mnTrackScaleLevel;
        float mTrackViewCos;
        float mTrackProjX;
        float mTrackProjY;
        int mnTrackScaleLevelR;
        float mTrackViewCosR;
        float mTrackProjXR;
        float mTrackProjYR;
        long unsigned int mnLastFrameSeen;
        int nObs;
        uint8_t mDescriptor[32];
        
    // For searchByProjection in PoseEstimation
    public:
        Eigen::Vector3f mWorldPos;
    };
}

#endif // CUDA_MAPPOINT_H