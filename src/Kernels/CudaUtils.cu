#include "Kernels/CudaUtils.h"

int CudaUtils::nFeatures_with_th;
int CudaUtils::nLevels; 
bool CudaUtils::isMonocular;  
float CudaUtils::scaleFactor;  
int CudaUtils::nCols;
int CudaUtils::nRows;
int CudaUtils::keypointsPerCell = 20;
int CudaUtils::maxNumOfMapPoints = 16000;
float* CudaUtils::d_mvScaleFactors;
int CudaUtils::ORBmatcher_TH_HIGH = 100;
int CudaUtils::ORBmatcher_TH_LOW = 50;
int CudaUtils::ORBmatcher_HISTO_LENGTH = 30;
bool CudaUtils::cameraIsFisheye;

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << msg << ": " << cudaGetErrorString(err) << ", status code: " << err << std::endl;
        exit(EXIT_FAILURE);
    }
}

void CudaUtils::loadSetting(int _nFeatures, int _nLevels, bool _isMonocular, float _scaleFactor, int _nCols, int _nRows, bool _cameraIsFisheye){
    nFeatures_with_th = _nFeatures + N_FEATURES_TH;
    nLevels = _nLevels;
    isMonocular = _isMonocular;
    scaleFactor = _scaleFactor;
    nCols = _nCols;
    nRows = _nRows;
    cameraIsFisheye = _cameraIsFisheye;
    std::vector<float> h_mvScaleFactors(nLevels);
    h_mvScaleFactors[0]=1.0f;
    for(int i=1; i<nLevels; i++)
    {
        h_mvScaleFactors[i]=h_mvScaleFactors[i-1]*scaleFactor;
    }
    checkCudaError(cudaMalloc(&d_mvScaleFactors, h_mvScaleFactors.size() * sizeof(float)), "CudaUtils:: Failed to allocate memory for d_mvScaleFactors");
    checkCudaError(cudaMemcpy(d_mvScaleFactors, h_mvScaleFactors.data(), h_mvScaleFactors.size() * sizeof(float), cudaMemcpyHostToDevice),"CudaUtils:: Failed to initialize d_mvScaleFactors"); 
}

__device__ int DescriptorDistance(const uint8_t *a, const uint8_t *b) {
    const int32_t *pa = reinterpret_cast<const int32_t*>(a);
    const int32_t *pb = reinterpret_cast<const int32_t*>(b);

    int dist = 0;

    for (int i = 0; i < DESCRIPTOR_SIZE / 4; i++, pa++, pb++) {
        unsigned int v = *pa ^ *pb;
        v = v - ((v >> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }

    return dist;
}
