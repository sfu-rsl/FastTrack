#include "Kernels/CudaWrappers/CudaMapPoint.h"

#ifdef TIME_MEASURMENT
#define TIMESTAMP_PRINT(msg) std::cout << "TimeStamp [CudaFrame]: " << msg << std::endl
#else
#define TIMESTAMP_PRINT(msg) do {} while (0)
#endif

namespace DATA_WRAPPER
{
    CudaMapPoint::CudaMapPoint() {
        isEmpty = true;
    }

    CudaMapPoint::CudaMapPoint(ORB_SLAM3::MapPoint* mp) {
        isEmpty = false;
        mnId = mp->mnId;
        mbTrackInView = mp->mbTrackInView;
        mbTrackInViewR = mp->mbTrackInViewR;
        mTrackDepth = mp->mTrackDepth;
        mnTrackScaleLevel = mp->mnTrackScaleLevel;
        mTrackViewCos = mp->mTrackViewCos;
        mTrackProjX = mp->mTrackProjX;
        mTrackProjY = mp->mTrackProjY;
        mnTrackScaleLevelR = mp->mnTrackScaleLevelR;
        mTrackViewCosR = mp->mTrackViewCosR;
        mTrackProjXR = mp->mTrackProjXR;
        mTrackProjYR = mp->mTrackProjYR;
        mnLastFrameSeen = mp->mnLastFrameSeen;
        nObs = mp->Observations();
        const cv::Mat& descriptor = mp->GetDescriptor();
        std::memcpy(mDescriptor, descriptor.ptr<uint8_t>(0), descriptor.cols * sizeof(uint8_t));
        mWorldPos = mp->GetWorldPos();
    }
}