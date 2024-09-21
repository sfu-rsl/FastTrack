#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <vector>
#include <opencv2/opencv.hpp>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#else
#include <cuda_runtime_api.h>
#endif

// Flag to activate the measurement of time in each kernel.
// #define REGISTER_STATS

#define N_FEATURES_TH 20
#define DESCRIPTOR_SIZE 32

class CudaUtils {
    public:
        static void loadSetting(int _nFeatures, int _nLevels, bool _isMonocular, float _scaleFactor, int _nCols, int _nRows, bool _cameraIsFisheye);

    public:
        static int nFeatures_with_th;
        static int nLevels; 
        static bool isMonocular; 
        static float scaleFactor; 
        static int nCols;
        static int nRows;
        static int keypointsPerCell;
        static int maxNumOfMapPoints;
        static int ORBmatcher_TH_HIGH;
        static int ORBmatcher_TH_LOW;
        static int ORBmatcher_HISTO_LENGTH;
        static float* d_mvScaleFactors;
        static bool cameraIsFisheye;
};

void checkCudaError(cudaError_t err, const char* msg);
__device__ int DescriptorDistance(const uint8_t *a, const uint8_t *b);

#endif // CUDA_UTILS_H