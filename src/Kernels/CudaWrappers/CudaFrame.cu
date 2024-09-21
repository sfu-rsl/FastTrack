#include "Kernels/CudaWrappers/CudaFrame.h"
#include <cstdio>
#include <vector>

// #define DEBUG
// #define TIME_MEASURMENT

#ifdef DEBUG
#define DEBUG_PRINT(msg) std::cout << "Debug [CudaFrame]: " << msg << std::endl
#else
#define DEBUG_PRINT(msg) do {} while (0)
#endif

#ifdef TIME_MEASURMENT
#define TIMESTAMP_PRINT(msg) std::cout << "TimeStamp [CudaFrame]: " << msg << std::endl
#else
#define TIMESTAMP_PRINT(msg) do {} while (0)
#endif

namespace DATA_WRAPPER
{
    void CudaFrame::initializeMemory(){
        DEBUG_PRINT("Allocating GPU memory For Frame...");

        int nFeatures = CudaUtils::nFeatures_with_th;
        
        int nLevels = CudaUtils::nLevels;
        
        bool cameraIsFisheye = CudaUtils::cameraIsFisheye;

        if (cameraIsFisheye) {
            checkCudaError(cudaMalloc((void**)&mvpMapPoints, 2 * nFeatures * sizeof(CudaMapPoint)), "Frame::failed to allocate memory for mvpMapPoints");
        } else {
            checkCudaError(cudaMalloc((void**)&mvpMapPoints, nFeatures * sizeof(CudaMapPoint)), "Frame::failed to allocate memory for mvpMapPoints");
        }

        checkCudaError(cudaMalloc((void**)&mvuRight, nFeatures * sizeof(float)), "Frame::failed to allocate memory for mvuRight");

        checkCudaError(cudaMalloc((void**)&mvKeys, nFeatures * sizeof(CudaKeyPoint)), "Frame::failed to allocate memory for mvKeys");
        
        checkCudaError(cudaMalloc((void**)&mvKeysRight, nFeatures * sizeof(CudaKeyPoint)), "Frame::failed to allocate memory for mvKeysRight");
        
        checkCudaError(cudaMalloc((void**)&mvKeysUn, nFeatures * sizeof(CudaKeyPoint)), "Frame::failed to allocate memory for mvKeysUn"); 
        
        if (cameraIsFisheye) {
            checkCudaError(cudaMalloc((void**)&mDescriptors, 2 * nFeatures * DESCRIPTOR_SIZE * sizeof(uint8_t)), "Frame::failed to allocate memory for mDescriptors");
        } else {
            checkCudaError(cudaMalloc((void**)&mDescriptors, nFeatures * DESCRIPTOR_SIZE * sizeof(uint8_t)), "Frame::failed to allocate memory for mDescriptors");
        }

        if (cameraIsFisheye) {
            checkCudaError(cudaMalloc((void**)&mvbOutlier, 2 * nFeatures * sizeof(uint8_t)), "Frame::failed to allocate memory for mvbOutlier");
        } else {
            checkCudaError(cudaMalloc((void**)&mvbOutlier, nFeatures * sizeof(uint8_t)), "Frame::failed to allocate memory for mvbOutlier");
        }
    }

    CudaFrame::CudaFrame() {
        initializeMemory();
    }

    void CudaFrame::setMvKeys(CudaKeyPoint* const &_mvKeys) {
         mvKeys = _mvKeys; 
         mvKeysIsOnGpu = true;
    }
    
    void CudaFrame::setMvKeysRight(CudaKeyPoint* const &_mvKeysRight) {
        mvKeysRight = _mvKeysRight; 
        mvKeysRightIsOnGpu = true;
    }

    void CudaFrame::setMDescriptors(uint8_t* const &_mDescriptors) { 
        mDescriptors = _mDescriptors; 
        mDescriptorsIsOnGpu = true;
    }

    void CudaFrame::setMemory(const ORB_SLAM3::Frame &F) {
        DEBUG_PRINT("Filling CudaFrame Memory With Frame Data...");

        mnId = F.mnId;
        N = F.N;
        Nleft = F.Nleft;
        mnMinX = F.mnMinX;
        mnMinY = F.mnMinY;
        mnMaxX = F.mnMaxX;
        mnMaxY = F.mnMaxY;
        mfGridElementWidthInv = F.mfGridElementWidthInv;
        mfGridElementHeightInv = F.mfGridElementHeightInv;
        mbf = F.mbf;
        mvKeys_size = F.mvKeys.size();
        mvKeysRight_size = F.mvKeysRight.size();
        mvKeysUn_size = F.mvKeysUn.size();
        mvpMapPoints_size = F.mvpMapPoints.size();
        mvuRight_size = F.mvuRight.size();
        mDescriptor_rows = F.mDescriptors.rows;
        mvbOutlier_size = F.mvbOutlier.size();

        checkCudaError(cudaMemcpy(mvuRight, F.mvuRight.data(), mvuRight_size * sizeof(float), cudaMemcpyHostToDevice), "CudaFrame:: Failed to copy mvuRight to gpu");
        
        if (!mDescriptorsIsOnGpu) {
            checkCudaError(cudaMemcpy((void*) mDescriptors, F.mDescriptors.data,  F.mDescriptors.rows * DESCRIPTOR_SIZE * sizeof(uint8_t), cudaMemcpyHostToDevice), "CudaFrame:: Failed to copy mDescriptors to gpu"); 
        } 
        
        if (!mvKeysIsOnGpu) {
            std::vector<CudaKeyPoint> tmp_mvKeys(mvKeys_size);
            for (int i = 0; i < mvKeys_size; ++i){
                tmp_mvKeys[i].ptx = F.mvKeys[i].pt.x;
                tmp_mvKeys[i].pty = F.mvKeys[i].pt.y;
                tmp_mvKeys[i].octave = F.mvKeys[i].octave;
            }
            checkCudaError(cudaMemcpy((void*) mvKeys, tmp_mvKeys.data(), mvKeys_size * sizeof(CudaKeyPoint), cudaMemcpyHostToDevice), "CudaFrame:: Failed to copy mvKeys to gpu");
        }

        if (!mvKeysRightIsOnGpu) {
            std::vector<CudaKeyPoint> tmp_mvKeysRight(mvKeysRight_size);        
            for (int i = 0; i < mvKeysRight_size; ++i){
                tmp_mvKeysRight[i].ptx = F.mvKeysRight[i].pt.x;
                tmp_mvKeysRight[i].pty = F.mvKeysRight[i].pt.y;
                tmp_mvKeysRight[i].octave = F.mvKeysRight[i].octave;
            }
            checkCudaError(cudaMemcpy((void*) mvKeysRight, tmp_mvKeysRight.data(), mvKeysRight_size * sizeof(CudaKeyPoint), cudaMemcpyHostToDevice), "CudaFrame:: Failed to copy mvKeysRight to gpu");
        }

        std::vector<CudaMapPoint> tmp_mvpMapPoints(mvpMapPoints_size);
        for (int i = 0; i < mvpMapPoints_size; ++i) {
                if (F.mvpMapPoints[i]) {
                    CudaMapPoint cuda_mp(F.mvpMapPoints[i]);
                    tmp_mvpMapPoints[i] = cuda_mp;
                } else {
                    CudaMapPoint cuda_mp;
                    tmp_mvpMapPoints[i] = cuda_mp;            
                }
        }
        checkCudaError(cudaMemcpy(mvpMapPoints, tmp_mvpMapPoints.data(), tmp_mvpMapPoints.size() * sizeof(CudaMapPoint), cudaMemcpyHostToDevice), "CudaFrame:: Failed to copy mvpMapPoints to gpu");
        

        std::vector<CudaKeyPoint> tmp_mvKeysUn(mvKeysUn_size);   
        for (int i = 0; i < mvKeysUn_size; ++i){
            tmp_mvKeysUn[i].ptx = F.mvKeysUn[i].pt.x;
            tmp_mvKeysUn[i].pty = F.mvKeysUn[i].pt.y;
            tmp_mvKeysUn[i].octave = F.mvKeysUn[i].octave;
        }
        checkCudaError(cudaMemcpy(mvKeysUn, tmp_mvKeysUn.data(), mvKeysUn_size * sizeof(CudaKeyPoint), cudaMemcpyHostToDevice), "CudaFrame:: Failed to copy mvKeysUn to gpu");

        int keypoints_per_cell = CudaUtils::keypointsPerCell;
        for (int i = 0; i < FRAME_GRID_COLS; ++i) {
            for (int j = 0; j < FRAME_GRID_ROWS; ++j) {
                size_t num_keypoints = F.mGrid[i][j].size();
                if (num_keypoints > 0) {
                    std::memcpy(&flatMGrid[(i * FRAME_GRID_ROWS + j) * keypoints_per_cell], F.mGrid[i][j].data(), num_keypoints * sizeof(std::size_t));
                }
                flatMGrid_size[i * FRAME_GRID_ROWS + j] = num_keypoints;
            }
        }

        if (!CudaUtils::cameraIsFisheye) {
            for (int i = 0; i < FRAME_GRID_COLS; ++i) {
                for (int j = 0; j < FRAME_GRID_ROWS; ++j) {
                    size_t num_keypoints = F.mGridRight[i][j].size();
                    if (num_keypoints > 0) {
                        std::memcpy(&flatMGridRight[(i * FRAME_GRID_ROWS + j) * KEYPOINTS_PER_CELL], F.mGridRight[i][j].data(), num_keypoints * sizeof(std::size_t));
                    }
                    flatMGridRight_size[i * FRAME_GRID_ROWS + j] = num_keypoints;
                }
            }
        }

        std::vector<uint8_t> intVec;
        intVec.resize(F.mvbOutlier.size());
        for (int i = 0; i < F.mvbOutlier.size(); ++i) {
            intVec[i] = F.mvbOutlier[i] ? 1 : 0;
        }
        checkCudaError(cudaMemcpy(mvbOutlier, intVec.data(),  intVec.size() * sizeof(uint8_t), cudaMemcpyHostToDevice), "CudaFrame:: Failed to copy mvbOutlier to gpu");

        std::memcpy(mpCamera_mvParameters, F.mpCamera->getParameters().data(), F.mpCamera->getParameters().size() * sizeof(float));
        
        mTrl = F.GetRelativePoseTrl().matrix();

        checkCudaError(cudaDeviceSynchronize(), "[cudaFrame:] failed to set memory");  

    }

    void CudaFrame::freeMemory(){
        DEBUG_PRINT("Freeing GPU Memory For Frame...");
        checkCudaError(cudaFree(mvpMapPoints),"Failed to free frame memory: mvpMapPoints");
        checkCudaError(cudaFree(mvuRight),"Failed to free frame memory: mvuRight");
        if (!mDescriptorsIsOnGpu)
            checkCudaError(cudaFree((void*) mDescriptors),"Failed to free frame memory: mDescriptors");
        if (!mvKeysIsOnGpu)   
            checkCudaError(cudaFree((void*) mvKeys),"Failed to free frame memory: mvKeys");
        if (!mvKeysRightIsOnGpu)
            checkCudaError(cudaFree((void*) mvKeysRight),"Failed to free frame memory: mvKeysRight");
        checkCudaError(cudaFree(mvKeysUn),"Failed to free frame memory: mvKeysUn");
        checkCudaError(cudaFree(mvbOutlier),"Failed to free frame memory: mvbOutlier");
    }
}