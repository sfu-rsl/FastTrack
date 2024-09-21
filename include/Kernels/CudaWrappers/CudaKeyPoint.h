#ifndef CUDA_KEYPOINT_H
#define CUDA_KEYPOINT_H

#include <opencv2/opencv.hpp>

namespace DATA_WRAPPER {

// class CudaKeyPoint {
//     public:
//         CudaKeyPoint();
//         CudaKeyPoint(int _oct, float _ptx, float _pty) : octave(_oct), ptx(_ptx), pty(_pty) {}

//     public:
//         int octave;
//         float ptx;
//         float pty;
//     };

    struct CudaKeyPoint {
        float ptx;
        float pty;
        int octave;
    };
}

#endif // CUDA_KEYPOINT_H