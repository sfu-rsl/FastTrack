#include "Kernels/SearchLocalPointsKernel.h"
#include <omp.h>
#include <memory.h>
#include <csignal> 


void SearchLocalPointsKernel::initialize() {
    if (memory_is_initialized) {
        return;
    }

    checkCudaError(cudaMalloc(&d_frame, sizeof(DATA_WRAPPER::CudaFrame)), "Failed to allocate memory for d_frame");
    
    checkCudaError(cudaMallocHost(&h_isEmpty, MAX_NUM_MAPPOINTS * sizeof(bool)), "Failed to allocate memory for h_isEmpty");
    checkCudaError(cudaMallocHost(&h_mbTrackInView, MAX_NUM_MAPPOINTS * sizeof(bool)), "Failed to allocate memory for h_mbTrackInView");
    checkCudaError(cudaMallocHost(&h_mbTrackInViewR, MAX_NUM_MAPPOINTS * sizeof(bool)), "Failed to allocate memory for h_mbTrackInViewR");
    checkCudaError(cudaMallocHost(&h_mTrackDepth, MAX_NUM_MAPPOINTS * sizeof(float)), "Failed to allocate memory for h_mTrackDepth");
    checkCudaError(cudaMallocHost(&h_mnTrackScaleLevel, MAX_NUM_MAPPOINTS * sizeof(int)), "Failed to allocate memory for h_mnTrackScaleLevel");
    checkCudaError(cudaMallocHost(&h_mTrackViewCos, MAX_NUM_MAPPOINTS * sizeof(float)), "Failed to allocate memory for h_mTrackViewCos");
    checkCudaError(cudaMallocHost(&h_mTrackProjX, MAX_NUM_MAPPOINTS * sizeof(float)), "Failed to allocate memory for h_mTrackProjX");
    checkCudaError(cudaMallocHost(&h_mTrackProjY, MAX_NUM_MAPPOINTS * sizeof(float)), "Failed to allocate memory for h_mTrackProjY");
    checkCudaError(cudaMallocHost(&h_mnTrackScaleLevelR, MAX_NUM_MAPPOINTS * sizeof(int)), "Failed to allocate memory for h_mnTrackScaleLevelR");
    checkCudaError(cudaMallocHost(&h_mTrackViewCosR, MAX_NUM_MAPPOINTS * sizeof(float)), "Failed to allocate memory for h_mTrackViewCosR");
    checkCudaError(cudaMallocHost(&h_mTrackProjXR, MAX_NUM_MAPPOINTS * sizeof(float)), "Failed to allocate memory for h_mTrackProjXR");
    checkCudaError(cudaMallocHost(&h_mTrackProjYR, MAX_NUM_MAPPOINTS * sizeof(float)), "Failed to allocate memory for h_mTrackProjYR");
    checkCudaError(cudaMallocHost(&h_mDescriptor, MAX_NUM_MAPPOINTS * DESCRIPTOR_SIZE * sizeof(uint8_t)), "Failed to allocate memory for h_mDescriptor");

    checkCudaError(cudaMalloc((void**)&d_isEmpty, MAX_NUM_MAPPOINTS * sizeof(bool)), "Failed to allocate memory for d_isEmpty");   
    checkCudaError(cudaMalloc((void**)&d_mbTrackInView, MAX_NUM_MAPPOINTS * sizeof(bool)), "Failed to allocate memory for d_mbTrackInView");
    checkCudaError(cudaMalloc((void**)&d_mbTrackInViewR, MAX_NUM_MAPPOINTS * sizeof(bool)), "Failed to allocate memory for d_mbTrackInViewR");
    checkCudaError(cudaMalloc((void**)&d_mTrackDepth, MAX_NUM_MAPPOINTS * sizeof(float)), "Failed to allocate memory for d_mTrackDepth");
    checkCudaError(cudaMalloc((void**)&d_mnTrackScaleLevel, MAX_NUM_MAPPOINTS * sizeof(int)), "Failed to allocate memory for d_mnTrackScaleLevel");
    checkCudaError(cudaMalloc((void**)&d_mTrackViewCos, MAX_NUM_MAPPOINTS * sizeof(float)), "Failed to allocate memory for d_mTrackViewCos");
    checkCudaError(cudaMalloc((void**)&d_mTrackProjX, MAX_NUM_MAPPOINTS * sizeof(float)), "Failed to allocate memory for d_mTrackProjX");
    checkCudaError(cudaMalloc((void**)&d_mTrackProjY, MAX_NUM_MAPPOINTS * sizeof(float)), "Failed to allocate memory for d_mTrackProjY");
    checkCudaError(cudaMalloc((void**)&d_mnTrackScaleLevelR, MAX_NUM_MAPPOINTS * sizeof(int)), "Failed to allocate memory for d_mnTrackScaleLevelR");
    checkCudaError(cudaMalloc((void**)&d_mTrackViewCosR, MAX_NUM_MAPPOINTS * sizeof(float)), "Failed to allocate memory for d_mTrackViewCosR");
    checkCudaError(cudaMalloc((void**)&d_mTrackProjXR, MAX_NUM_MAPPOINTS * sizeof(float)), "Failed to allocate memory for d_mTrackProjXR");
    checkCudaError(cudaMalloc((void**)&d_mTrackProjYR, MAX_NUM_MAPPOINTS * sizeof(float)), "Failed to allocate memory for d_mTrackProjYR");
    checkCudaError(cudaMalloc((void**)&d_mDescriptor, MAX_NUM_MAPPOINTS * DESCRIPTOR_SIZE * sizeof(uint8_t)), "Failed to allocate memory for d_mDescriptor");   

    checkCudaError(cudaMallocHost(&h_bestLevel, MAX_NUM_MAPPOINTS * sizeof(int)), "Failed to allocate memory for h_bestLevel");
    checkCudaError(cudaMallocHost(&h_bestLevel2, MAX_NUM_MAPPOINTS * sizeof(int)), "Failed to allocate memory for h_bestLevel2");   
    checkCudaError(cudaMallocHost(&h_bestDist, MAX_NUM_MAPPOINTS * sizeof(int)), "Failed to allocate memory for h_bestDist");
    checkCudaError(cudaMallocHost(&h_bestDist2, MAX_NUM_MAPPOINTS * sizeof(int)), "Failed to allocate memory for h_bestDist2");
    checkCudaError(cudaMallocHost(&h_bestIdx, MAX_NUM_MAPPOINTS * sizeof(int)), "Failed to allocate memory for h_bestIdx");      
    checkCudaError(cudaMalloc((void**)&d_bestLevel, MAX_NUM_MAPPOINTS * sizeof(int)), "Failed to allocate memory for d_bestLevel");
    checkCudaError(cudaMalloc((void**)&d_bestLevel2, MAX_NUM_MAPPOINTS * sizeof(int)), "Failed to allocate memory for d_bestLevel2");   
    checkCudaError(cudaMalloc((void**)&d_bestDist, MAX_NUM_MAPPOINTS * sizeof(int)), "Failed to allocate memory for d_bestDist");
    checkCudaError(cudaMalloc((void**)&d_bestDist2, MAX_NUM_MAPPOINTS * sizeof(int)), "Failed to allocate memory for d_bestDist2");
    checkCudaError(cudaMalloc((void**)&d_bestIdx, MAX_NUM_MAPPOINTS * sizeof(int)), "Failed to allocate memory for d_bestIdx");  

    checkCudaError(cudaMallocHost(&h_bestLevelR, MAX_NUM_MAPPOINTS * sizeof(int)), "Failed to allocate memory for h_bestLevelR");
    checkCudaError(cudaMallocHost(&h_bestLevelR2, MAX_NUM_MAPPOINTS * sizeof(int)), "Failed to allocate memory for h_bestLevelR2");   
    checkCudaError(cudaMallocHost(&h_bestDistR, MAX_NUM_MAPPOINTS * sizeof(int)), "Failed to allocate memory for h_bestDistR");
    checkCudaError(cudaMallocHost(&h_bestDistR2, MAX_NUM_MAPPOINTS * sizeof(int)), "Failed to allocate memory for h_bestDistR2");
    checkCudaError(cudaMallocHost(&h_bestIdxR, MAX_NUM_MAPPOINTS * sizeof(int)), "Failed to allocate memory for h_bestIdxR");      
    checkCudaError(cudaMalloc((void**)&d_bestLevelR, MAX_NUM_MAPPOINTS * sizeof(int)), "Failed to allocate memory for d_bestLevelR");
    checkCudaError(cudaMalloc((void**)&d_bestLevelR2, MAX_NUM_MAPPOINTS * sizeof(int)), "Failed to allocate memory for d_bestLevelR2");   
    checkCudaError(cudaMalloc((void**)&d_bestDistR, MAX_NUM_MAPPOINTS * sizeof(int)), "Failed to allocate memory for d_bestDistR");
    checkCudaError(cudaMalloc((void**)&d_bestDistR2, MAX_NUM_MAPPOINTS * sizeof(int)), "Failed to allocate memory for d_bestDistR2");
    checkCudaError(cudaMalloc((void**)&d_bestIdxR, MAX_NUM_MAPPOINTS * sizeof(int)), "Failed to allocate memory for d_bestIdxR");  

    memory_is_initialized = true;
}

void SearchLocalPointsKernel::setFrame(DATA_WRAPPER::CudaFrame* cudaFrame) {
    checkCudaError(cudaMemcpy(d_frame, cudaFrame, sizeof(DATA_WRAPPER::CudaFrame), cudaMemcpyHostToDevice), "Failed to copy Frame to device");
}

__global__ void searchByProjectionKernel(DATA_WRAPPER::CudaFrame* d_frame, 
                                bool *d_isEmpty,
                                bool *d_mbTrackInView,
                                bool *d_mbTrackInViewR,
                                float *d_mTrackDepth,
                                int *d_mnTrackScaleLevel,
                                float *d_mTrackViewCos,
                                float *d_mTrackProjX,
                                float *d_mTrackProjY,
                                int *d_mnTrackScaleLevelR,
                                float *d_mTrackViewCosR,
                                float *d_mTrackProjXR,
                                float *d_mTrackProjYR,
                                uint8_t *d_mDescriptor,
                                float* d_mvScaleFactors, int numPoints, const float th,
                                int* d_bestLevel, int* d_bestLevel2, int* d_bestDist, int* d_bestDist2, int* d_bestIdx,
                                int* d_bestLevelR, int* d_bestLevelR2, int* d_bestDistR, int* d_bestDistR2, int* d_bestIdxR) {

    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < numPoints) {
        d_bestDist[idx] = 256;
        d_bestLevel[idx] = -1;
        d_bestDist2[idx] = 256;
        d_bestLevel2[idx] = -1;
        d_bestIdx[idx] = -1;
        d_bestDistR[idx] = 256;
        d_bestLevelR[idx] = -1;
        d_bestDistR2[idx] = 256;
        d_bestLevelR2[idx] = -1;
        d_bestIdxR[idx] = -1;
        if (!d_isEmpty[idx]) { 
            if (d_mbTrackInView[idx]) {
                int nPredictedLevel = d_mnTrackScaleLevel[idx];
                float x = d_mTrackProjX[idx];
                float y = d_mTrackProjY[idx];
                float r = (d_mTrackViewCos[idx] > 0.998) ? 2.5 : 4.0;
                r = r * th;
                bool bRight = false;

                const int minLevel = nPredictedLevel-1;
                const int maxLevel = nPredictedLevel;

                // ## Finding The Closest KeyPoint Attributes ##
                const uint8_t* MPdescriptor = &d_mDescriptor[idx*DESCRIPTOR_SIZE];
                int bestDist=256;
                int bestLevel= -1;
                int bestDist2= 256;
                int bestLevel2 = -1;
                int bestIdx = -1;

                // ## GetFeaturesInArea Function ##
                float factorX = r;
                float factorY = r;

                const int nMinCellX = max(0, (int)floor((x - d_frame->mnMinX - factorX) * d_frame->mfGridElementWidthInv));
                if(nMinCellX >= FRAME_GRID_COLS) {
                    return;
                }

                const int nMaxCellX = min((int)FRAME_GRID_COLS-1, (int)ceil((x - d_frame->mnMinX + factorX) * d_frame->mfGridElementWidthInv));
                if(nMaxCellX < 0) {
                    return;
                }

                const int nMinCellY = max(0, (int)floor((y - d_frame->mnMinY - factorY) * d_frame->mfGridElementHeightInv));
                if(nMinCellY >= FRAME_GRID_ROWS) {
                    return;
                }

                const int nMaxCellY = min((int)FRAME_GRID_ROWS-1, (int)ceil((y - d_frame->mnMinY + factorY) * d_frame->mfGridElementHeightInv));
                if(nMaxCellY < 0) {
                    return;
                }

                const bool bCheckLevels = (minLevel>0) || (maxLevel>=0);

                for(int ix = nMinCellX; ix<=nMaxCellX; ix++) {
                    for(int iy = nMinCellY; iy<=nMaxCellY; iy++) {
                        std::size_t* vCell;
                        int vCell_size;
                        if (!bRight) {
                            vCell = &d_frame->flatMGrid[ix * FRAME_GRID_ROWS * KEYPOINTS_PER_CELL + iy * KEYPOINTS_PER_CELL];
                            vCell_size = d_frame->flatMGrid_size[ix * FRAME_GRID_ROWS + iy];
                        } else {
                            vCell = &d_frame->flatMGridRight[ix * FRAME_GRID_ROWS * KEYPOINTS_PER_CELL + iy * KEYPOINTS_PER_CELL];
                            vCell_size = d_frame->flatMGridRight_size[ix * FRAME_GRID_ROWS + iy];
                        }

                        if(vCell_size == 0) {
                            continue;
                        }
                        for(size_t j=0, jend=vCell_size; j<jend; j++) {
                            

                            const DATA_WRAPPER::CudaKeyPoint &kpUn = (d_frame->Nleft == -1) ? d_frame->mvKeysUn[vCell[j]]
                                                                    : (!bRight) ? d_frame->mvKeys[vCell[j]]
                                                                                : d_frame->mvKeysRight[vCell[j]];
                
                            if (bCheckLevels) {
                                if(kpUn.octave<minLevel)
                                    continue;
                                if(maxLevel>=0)
                                    if(kpUn.octave>maxLevel)
                                        continue;
                            }

                            const float distx = kpUn.ptx-x;
                            const float disty = kpUn.pty-y;

                            if (fabs(distx)<factorX && fabs(disty)<factorY) {
                                
                                if (!d_frame->mvpMapPoints[vCell[j]].isEmpty) {
                                    if(d_frame->mvpMapPoints[vCell[j]].nObs > 0) {
                                        continue;
                                    }
                                }

                                if (d_frame->Nleft == -1 && d_frame->mvuRight[vCell[j]]>0) {
                                    const float er = fabs(d_mTrackProjXR[idx] - d_frame->mvuRight[vCell[j]]);
                                    if(er > r * d_mvScaleFactors[nPredictedLevel]) {
                                        continue;
                                    }
                                }

                                const uint8_t* d = &d_frame->mDescriptors[vCell[j] * DESCRIPTOR_SIZE];
                    
                                int dist = DescriptorDistance(MPdescriptor, d);                          

                                if (dist<bestDist) {
                                    bestDist2=bestDist;
                                    bestDist=dist;
                                    bestLevel2 = bestLevel;
                                    bestLevel = (d_frame->Nleft == -1) ? d_frame->mvKeysUn[vCell[j]].octave
                                                                    : (vCell[j] < d_frame->Nleft) ? d_frame->mvKeys[vCell[j]].octave
                                                                                    : d_frame->mvKeysRight[vCell[j] - d_frame->Nleft].octave;
                            
                                    bestIdx=vCell[j];
                                }
                                else if (dist<bestDist2) {
                                    bestLevel2 = (d_frame->Nleft == -1) ? d_frame->mvKeysUn[vCell[j]].octave
                                                                    : (vCell[j] < d_frame->Nleft) ? d_frame->mvKeys[vCell[j]].octave
                                                                                    : d_frame->mvKeysRight[vCell[j] - d_frame->Nleft].octave;
                                            
                                    bestDist2=dist;
                                }
                            }
                        }
                    }
                }
                d_bestDist[idx] = bestDist;
                d_bestLevel[idx] = bestLevel;
                d_bestDist2[idx] = bestDist2;
                d_bestLevel2[idx] = bestLevel2;
                d_bestIdx[idx] = bestIdx;
            }

            if (d_frame->Nleft != -1 && d_mbTrackInViewR[idx]) {
                int nPredictedLevel = d_mnTrackScaleLevelR[idx];
                if (nPredictedLevel == -1) {
                    return;
                }
                float x = d_mTrackProjXR[idx];
                float y = d_mTrackProjYR[idx];
                float r = (d_mTrackViewCosR[idx] > 0.998) ? 2.5 : 4.0;
                r = r * d_mvScaleFactors[nPredictedLevel];
                bool bRight = true;
            
                const int minLevel = nPredictedLevel-1;
                const int maxLevel = nPredictedLevel;

                // ## Finding The Closest KeyPoint Attributes ##
                const uint8_t* MPdescriptor = &d_mDescriptor[idx*DESCRIPTOR_SIZE];
                int bestDistR=256;
                int bestLevelR= -1;
                int bestDistR2= 256;
                int bestLevelR2 = -1;
                int bestIdxR = -1;

                // ## GetFeaturesInArea Function ##
                float factorX = r;
                float factorY = r;

                const int nMinCellX = max(0, (int)floor((x - d_frame->mnMinX - factorX) * d_frame->mfGridElementWidthInv));
                if(nMinCellX >= FRAME_GRID_COLS) {
                    return;
                }

                const int nMaxCellX = min((int)FRAME_GRID_COLS-1, (int)ceil((x - d_frame->mnMinX + factorX) * d_frame->mfGridElementWidthInv));
                if(nMaxCellX < 0) {
                    return;
                }

                const int nMinCellY = max(0, (int)floor((y - d_frame->mnMinY - factorY) * d_frame->mfGridElementHeightInv));
                if(nMinCellY >= FRAME_GRID_ROWS) {
                    return;
                }

                const int nMaxCellY = min((int)FRAME_GRID_ROWS-1, (int)ceil((y - d_frame->mnMinY + factorY) * d_frame->mfGridElementHeightInv));
                if(nMaxCellY < 0) {
                    return;
                }

                const bool bCheckLevels = (minLevel>0) || (maxLevel>=0);

                for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
                {
                    for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
                    {
                        std::size_t* vCell;
                        int vCell_size;
                        if (!bRight) {
                            vCell = &d_frame->flatMGrid[ix * FRAME_GRID_ROWS * KEYPOINTS_PER_CELL + iy * KEYPOINTS_PER_CELL];
                            vCell_size = d_frame->flatMGrid_size[ix * FRAME_GRID_ROWS + iy];
                        } else {
                            vCell = &d_frame->flatMGridRight[ix * FRAME_GRID_ROWS * KEYPOINTS_PER_CELL + iy * KEYPOINTS_PER_CELL];
                            vCell_size = d_frame->flatMGridRight_size[ix * FRAME_GRID_ROWS + iy];
                        }
                        if(vCell_size == 0) {
                            continue;
                        }
                        for(size_t j=0, jend=vCell_size; j<jend; j++) {
                            
                            const DATA_WRAPPER::CudaKeyPoint &kpUn = (d_frame->Nleft == -1) ? d_frame->mvKeysUn[vCell[j]]
                                                                    : (!bRight) ? d_frame->mvKeys[vCell[j]]
                                                                                : d_frame->mvKeysRight[vCell[j]];
                            if(bCheckLevels)
                            {
                                if(kpUn.octave<minLevel)
                                    continue;
                                if(maxLevel>=0)
                                    if(kpUn.octave>maxLevel)
                                        continue;
                            }

                            const float distx = kpUn.ptx-x;
                            const float disty = kpUn.pty-y;

                            if(fabs(distx)<factorX && fabs(disty)<factorY) {

                                if(!d_frame->mvpMapPoints[vCell[j] + d_frame->Nleft].isEmpty) {
                                    if(d_frame->mvpMapPoints[vCell[j] + d_frame->Nleft].nObs > 0)
                                        continue;
                                }

                                const uint8_t* d = &d_frame->mDescriptors[(vCell[j] + d_frame->Nleft) * DESCRIPTOR_SIZE];
                    
                                int dist = DescriptorDistance(MPdescriptor, d);

                                if(dist<bestDistR)
                                {
                                    bestDistR2=bestDistR;
                                    bestDistR=dist;
                                    bestLevelR2 = bestLevelR;
                                    bestLevelR = d_frame->mvKeysRight[vCell[j]].octave;
                                    bestIdxR=vCell[j];
                                }
                                else if(dist<bestDistR2)
                                {
                                    bestLevelR2 = d_frame->mvKeysRight[vCell[j]].octave;        
                                    bestDistR2=dist;
                                }
                            }
                        }
                    }
                }
                d_bestDistR[idx] = bestDistR;
                d_bestLevelR[idx] = bestLevelR;
                d_bestDistR2[idx] = bestDistR2;
                d_bestLevelR2[idx] = bestLevelR2;
                d_bestIdxR[idx] = bestIdxR;
            }
        }
    }
}

void SearchLocalPointsKernel::launch(ORB_SLAM3::Frame &F, const vector<ORB_SLAM3::MapPoint*> &vmp, const float th, const bool bFarPoints, const float thFarPoints,
                        int* h_bestLevel, int* h_bestLevel2, int* h_bestDist, int* h_bestDist2, int* h_bestIdx,
                        int* h_bestLevelR, int* h_bestLevelR2, int* h_bestDistR, int* h_bestDistR2, int* h_bestIdxR){

#ifdef REGISTER_STATS
    std::chrono::steady_clock::time_point startTotal = std::chrono::steady_clock::now();
#endif
    
    if (!memory_is_initialized){
        initialize();
    }

    int numPoints = vmp.size();
    if(numPoints > MAX_NUM_MAPPOINTS) {
        cout << "[ERROR] SearchLocalPointsKernel::launchKernel: ] number of mappoints: " << numPoints << " is greater than MAX_NUM_MAPPOINTS: " << MAX_NUM_MAPPOINTS << "\n";
        raise(SIGSEGV);
    }

#ifdef REGISTER_STATS
    std::chrono::steady_clock::time_point startMapPointsWrap = std::chrono::steady_clock::now();
#endif

    #pragma omp parallel for
    for (int i = 0; i < numPoints; ++i) {
        ORB_SLAM3::MapPoint* pMP = vmp[i];
        if((!pMP->mbTrackInView && !pMP->mbTrackInViewR) || 
            (bFarPoints && pMP->mTrackDepth>thFarPoints) || 
            (pMP->isBad())){
            h_isEmpty[i] = true;
            continue;
        }
        h_isEmpty[i] = false;
        h_mbTrackInView[i] = pMP->mbTrackInView;
        h_mbTrackInViewR[i] = pMP->mbTrackInViewR;
        h_mTrackDepth[i] = pMP->mTrackDepth;
        h_mnTrackScaleLevel[i] = pMP->mnTrackScaleLevel;
        h_mTrackViewCos[i] = pMP->mTrackViewCos;
        h_mTrackProjX[i] = pMP->mTrackProjX;
        h_mTrackProjY[i] = pMP->mTrackProjY;
        h_mnTrackScaleLevelR[i] = pMP->mnTrackScaleLevelR;
        h_mTrackViewCosR[i] = pMP->mTrackViewCosR;
        h_mTrackProjXR[i] = pMP->mTrackProjXR;
        h_mTrackProjYR[i] = pMP->mTrackProjYR;
        std::memcpy(&h_mDescriptor[i*DESCRIPTOR_SIZE], pMP->GetDescriptor().data, DESCRIPTOR_SIZE * sizeof(uint8_t));
    }

#ifdef REGISTER_STATS
    std::chrono::steady_clock::time_point endMapPointsWrap = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point startMapPointsTransfer = std::chrono::steady_clock::now();
#endif

    cudaMemcpy(d_isEmpty, h_isEmpty, numPoints * sizeof(bool), cudaMemcpyHostToDevice); 
    cudaMemcpy(d_mbTrackInView, h_mbTrackInView, numPoints * sizeof(bool), cudaMemcpyHostToDevice); 
    cudaMemcpy(d_mbTrackInViewR, h_mbTrackInViewR, numPoints * sizeof(bool), cudaMemcpyHostToDevice); 
    cudaMemcpy(d_mTrackDepth, h_mTrackDepth, numPoints * sizeof(float), cudaMemcpyHostToDevice); 
    cudaMemcpy(d_mnTrackScaleLevel, h_mnTrackScaleLevel, numPoints * sizeof(int), cudaMemcpyHostToDevice); 
    cudaMemcpy(d_mTrackViewCos, h_mTrackViewCos, numPoints * sizeof(float), cudaMemcpyHostToDevice); 
    cudaMemcpy(d_mTrackProjX, h_mTrackProjX, numPoints * sizeof(float), cudaMemcpyHostToDevice); 
    cudaMemcpy(d_mTrackProjY, h_mTrackProjY, numPoints * sizeof(float), cudaMemcpyHostToDevice); 
    cudaMemcpy(d_mnTrackScaleLevelR, h_mnTrackScaleLevelR, numPoints * sizeof(int), cudaMemcpyHostToDevice); 
    cudaMemcpy(d_mTrackViewCosR, h_mTrackViewCosR, numPoints * sizeof(float), cudaMemcpyHostToDevice); 
    cudaMemcpy(d_mTrackProjXR, h_mTrackProjXR, numPoints * sizeof(float), cudaMemcpyHostToDevice); 
    cudaMemcpy(d_mTrackProjYR, h_mTrackProjYR, numPoints * sizeof(float), cudaMemcpyHostToDevice); 
    cudaMemcpy(d_mDescriptor, h_mDescriptor, numPoints * DESCRIPTOR_SIZE * sizeof(uint8_t), cudaMemcpyHostToDevice); 

#ifdef REGISTER_STATS
    std::chrono::steady_clock::time_point endMapPointsTransfer = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point startKernel = std::chrono::steady_clock::now();
#endif

    int blockSize = 256;
    int numBlocks = (numPoints + blockSize - 1) / blockSize;
    searchByProjectionKernel<<<numBlocks, blockSize>>>(d_frame, 
                                                        d_isEmpty,
                                                        d_mbTrackInView,
                                                        d_mbTrackInViewR,
                                                        d_mTrackDepth,
                                                        d_mnTrackScaleLevel,
                                                        d_mTrackViewCos,
                                                        d_mTrackProjX,
                                                        d_mTrackProjY,
                                                        d_mnTrackScaleLevelR,
                                                        d_mTrackViewCosR,
                                                        d_mTrackProjXR,
                                                        d_mTrackProjYR,
                                                        d_mDescriptor,
                                                        CudaUtils::d_mvScaleFactors, numPoints, th, 
                                                        d_bestLevel, d_bestLevel2, d_bestDist, d_bestDist2, d_bestIdx,
                                                        d_bestLevelR, d_bestLevelR2, d_bestDistR, d_bestDistR2, d_bestIdxR);
    checkCudaError(cudaDeviceSynchronize(), "[searchByProjectionKernel:] Kernel launch failed");  

#ifdef REGISTER_STATS
    std::chrono::steady_clock::time_point endKernel = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point startOutputTransfer = std::chrono::steady_clock::now();
#endif

    checkCudaError(cudaMemcpy(h_bestLevel, d_bestLevel, numPoints * sizeof(int), cudaMemcpyDeviceToHost), "Failed to copy d_bestLevel back to host");
    checkCudaError(cudaMemcpy(h_bestLevel2, d_bestLevel2, numPoints * sizeof(int), cudaMemcpyDeviceToHost), "Failed to copy d_bestLevel2 back to host");
    checkCudaError(cudaMemcpy(h_bestDist, d_bestDist, numPoints * sizeof(int), cudaMemcpyDeviceToHost), "Failed to copy d_bestDist back to host");
    checkCudaError(cudaMemcpy(h_bestDist2, d_bestDist2, numPoints * sizeof(int), cudaMemcpyDeviceToHost), "Failed to copy d_bestDist2 back to host");
    checkCudaError(cudaMemcpy(h_bestIdx, d_bestIdx, numPoints * sizeof(int), cudaMemcpyDeviceToHost), "Failed to copy d_bestIdx back to host");
    checkCudaError(cudaMemcpy(h_bestLevelR, d_bestLevelR, numPoints * sizeof(int), cudaMemcpyDeviceToHost), "Failed to copy d_bestLevelR back to host");
    checkCudaError(cudaMemcpy(h_bestLevelR2, d_bestLevelR2, numPoints * sizeof(int), cudaMemcpyDeviceToHost), "Failed to copy d_bestLevelR2 back to host");
    checkCudaError(cudaMemcpy(h_bestDistR, d_bestDistR, numPoints * sizeof(int), cudaMemcpyDeviceToHost), "Failed to copy d_bestDistR back to host");
    checkCudaError(cudaMemcpy(h_bestDistR2, d_bestDistR2, numPoints * sizeof(int), cudaMemcpyDeviceToHost), "Failed to copy d_bestDistR2 back to host");
    checkCudaError(cudaMemcpy(h_bestIdxR, d_bestIdxR, numPoints * sizeof(int), cudaMemcpyDeviceToHost), "Failed to copy d_bestIdxR back to host");

#ifdef REGISTER_STATS
    std::chrono::steady_clock::time_point endOutputTransfer = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point endTotal = std::chrono::steady_clock::now();

    double frameWrap = frame_wrap_time.back().second;
    double frameTransfer = frame_data_transfer_time.back().second;
    double mapPointsWrap = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(endMapPointsWrap - startMapPointsWrap).count();
    double mapPointsTransfer = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(endMapPointsTransfer - startMapPointsTransfer).count();
    double kernel = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(endKernel - startKernel).count();
    double outputTransfer = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(endOutputTransfer - startOutputTransfer).count();
    double total = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(endTotal - startTotal).count();

    data_transfer_time.emplace_back(F.mnId, frameTransfer + mapPointsTransfer + outputTransfer);
    input_data_transfer_time.emplace_back(F.mnId, frameTransfer + mapPointsTransfer);
    output_data_transfer_time.emplace_back(F.mnId, outputTransfer);
    mappoints_data_transfer_time.emplace_back(F.mnId, mapPointsTransfer);
    
    data_wrap_time.emplace_back(F.mnId, mapPointsWrap);
    mappoints_wrap_time.emplace_back(F.mnId, mapPointsWrap);
    
    kernel_exec_time.emplace_back(F.mnId, kernel);
    total_exec_time.emplace_back(F.mnId, total + frameWrap + frameTransfer);
#endif

    return;
}

void SearchLocalPointsKernel::shutdown(){
    if (!memory_is_initialized) 
        return;
    cudaFree(d_frame);
    cudaFreeHost(h_isEmpty);
    cudaFreeHost(h_mbTrackInView);
    cudaFreeHost(h_mbTrackInViewR);
    cudaFreeHost(h_mTrackDepth);
    cudaFreeHost(h_mnTrackScaleLevel);
    cudaFreeHost(h_mTrackViewCos);
    cudaFreeHost(h_mTrackProjX);
    cudaFreeHost(h_mTrackProjY);
    cudaFreeHost(h_mnTrackScaleLevelR);
    cudaFreeHost(h_mTrackViewCosR);
    cudaFreeHost(h_mTrackProjXR);
    cudaFreeHost(h_mTrackProjYR);
    cudaFreeHost(h_mDescriptor);
    cudaFreeHost(h_bestLevel);
    cudaFreeHost(h_bestLevel2);
    cudaFreeHost(h_bestDist);
    cudaFreeHost(h_bestDist2);
    cudaFreeHost(h_bestIdx);
    cudaFree(d_isEmpty);
    cudaFree(d_mbTrackInView);
    cudaFree(d_mbTrackInViewR);
    cudaFree(d_mTrackDepth);
    cudaFree(d_mnTrackScaleLevel);
    cudaFree(d_mTrackViewCos);
    cudaFree(d_mTrackProjX);
    cudaFree(d_mTrackProjY);
    cudaFree(d_mnTrackScaleLevelR);
    cudaFree(d_mTrackViewCosR);
    cudaFree(d_mTrackProjXR);
    cudaFree(d_mTrackProjYR);
    cudaFree(d_bestLevel);
    cudaFree(d_bestLevel2);
    cudaFree(d_bestDist);
    cudaFree(d_bestDist2);
    cudaFree(d_bestIdx);
}

void SearchLocalPointsKernel::saveStats(const string &file_path){

    string data_path = file_path + "/SearchLocalPointsKernel/";
    cout << "[SeachLocalPointsKernel:] writing stats data into file: " << data_path << '\n';
    if (mkdir(data_path.c_str(), 0755) == -1) {
        std::cerr << "[SearchLocalPointsKernel:] Error creating directory: " << strerror(errno) << std::endl;
    }

    ofstream myfile;

    myfile.open(data_path + "/total_exec_time.txt");
    for (const auto& p : total_exec_time) {
        myfile << p.first << ": " << p.second << std::endl;
    }
    myfile.close();

    myfile.open(data_path + "/data_wrap_time.txt");
    for (const auto& p : data_wrap_time) {
        myfile << p.first << ": " << p.second << std::endl;
    }
    myfile.close();

    myfile.open(data_path + "/data_transfer_time.txt");
    for (const auto& p : data_transfer_time) {
        myfile << p.first << ": " << p.second << std::endl;
    }
    myfile.close();

    myfile.open(data_path + "/kernel_exec_time.txt");
    for (const auto& p : kernel_exec_time) {
        myfile << p.first << ": " << p.second << std::endl;
    }
    myfile.close();

    myfile.open(data_path + "/input_data_transfer_time.txt");
    for (const auto& p : input_data_transfer_time) {
        myfile << p.first << ": " << p.second << std::endl;
    }
    myfile.close();

    myfile.open(data_path + "/output_data_transfer_time.txt");
    for (const auto& p : output_data_transfer_time) {
        myfile << p.first << ": " << p.second << std::endl;
    }
    myfile.close();

    myfile.open(data_path + "/mappoints_wrap_time.txt");
    for (const auto& p : mappoints_wrap_time) {
        myfile << p.first << ": " << p.second << std::endl;
    }
    myfile.close();

    myfile.open(data_path + "/frame_wrap_time.txt");
    for (const auto& p : frame_wrap_time) {
        myfile << p.first << ": " << p.second << std::endl;
    }
    myfile.close();

    myfile.open(data_path + "/frame_data_transfer_time.txt");
    for (const auto& p : frame_data_transfer_time) {
        myfile << p.first << ": " << p.second << std::endl;
    }
    myfile.close();

    myfile.open(data_path + "/mappoints_data_transfer_time.txt");
    for (const auto& p : mappoints_data_transfer_time) {
        myfile << p.first << ": " << p.second << std::endl;
    }
    myfile.close();
}