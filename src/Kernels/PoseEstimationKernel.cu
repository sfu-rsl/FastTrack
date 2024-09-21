#include "Kernels/PoseEstimationKernel.h"

void PoseEstimationKernel::initialize(){
    if (memory_is_initialized){
        return;
    }

    checkCudaError(cudaMalloc(&d_currentFrame, sizeof(DATA_WRAPPER::CudaFrame)), "Failed to allocate memory for d_currentFrame");

    checkCudaError(cudaMalloc(&d_lastFrame, sizeof(DATA_WRAPPER::CudaFrame)), "Failed to allocate memory for d_lastFrame");

    int Nfeatures = CudaUtils::nFeatures_with_th;
    if (!CudaUtils::isMonocular) {
        Nfeatures = 2 * Nfeatures;
    }

    checkCudaError(cudaMalloc((void**)&d_bestDist, Nfeatures * sizeof(int)), "Failed to allocate memory for d_bestDist");
    checkCudaError(cudaMalloc((void**)&d_bestIdx2, Nfeatures * sizeof(int)), "Failed to allocate memory for d_bestIdx2"); 
    checkCudaError(cudaMalloc((void**)&d_bestDistR, Nfeatures * sizeof(int)), "Failed to allocate memory for d_bestDistR");
    checkCudaError(cudaMalloc((void**)&d_bestIdxR2, Nfeatures * sizeof(int)), "Failed to allocate memory for d_bestIdxR2"); 

    memory_is_initialized = true;
}

void PoseEstimationKernel::setCurrentFrame(DATA_WRAPPER::CudaFrame* cudaFrame) {
    checkCudaError(cudaMemcpy(d_currentFrame, cudaFrame, sizeof(DATA_WRAPPER::CudaFrame), cudaMemcpyHostToDevice), "Failed to copy Current Frame to device");
}

void PoseEstimationKernel::setLastFrame(DATA_WRAPPER::CudaFrame* cudaFrame) {
    checkCudaError(cudaMemcpy(d_lastFrame, cudaFrame, sizeof(DATA_WRAPPER::CudaFrame), cudaMemcpyHostToDevice), "Failed to copy Last Frame to device");
}

__device__ Eigen::Vector2f cameraProject_KannalaBrandt8(float* mvParameters, const Eigen::Vector3f v3D) {
    const float x2_plus_y2 = v3D[0] * v3D[0] + v3D[1] * v3D[1]; 
    const float theta = atan2f(sqrtf(x2_plus_y2), v3D[2]);
    const float psi = atan2f(v3D[1], v3D[0]);

    const float theta2 = theta * theta;
    const float theta3 = theta * theta2;
    const float theta5 = theta3 * theta2;
    const float theta7 = theta5 * theta2;
    const float theta9 = theta7 * theta2;
    const float r = theta + mvParameters[4] * theta3 + mvParameters[5] * theta5
                        + mvParameters[6] * theta7 + mvParameters[7] * theta9;

    Eigen::Vector2f res;
    res[0] = mvParameters[0] * r * cos(psi) + mvParameters[2];
    res[1] = mvParameters[1] * r * sin(psi) + mvParameters[3];

    return res;
}

__device__ Eigen::Vector2f cameraProject_Pinhole(float* mvParameters, const Eigen::Vector3f v3D) {
    Eigen::Vector2f res;
    res[0] = mvParameters[0] * v3D[0] / v3D[2] + mvParameters[2];
    res[1] = mvParameters[1] * v3D[1] / v3D[2] + mvParameters[3];

    return res;
}

__global__ void searchByProjectionKernel(DATA_WRAPPER::CudaFrame *d_currentFrame, 
                                    DATA_WRAPPER::CudaFrame *d_lastFrame,
                                    float* d_mvScaleFactors, bool cameraIsFisheye,
                                    const float th, bool bForward, bool bBackward, 
                                    Eigen::Matrix4f transform_matrix,
                                    int *d_bestDist, int *d_bestIdx2, int *d_bestDistR, int *d_bestIdxR2) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < d_lastFrame->N){
        
        d_bestDist[idx] = 256;
        d_bestIdx2[idx] = -1;
        d_bestDistR[idx] = 256;
        d_bestIdxR2[idx] = -1;
        DATA_WRAPPER::CudaMapPoint pMP = d_lastFrame->mvpMapPoints[idx];
        
        if(!pMP.isEmpty) {

            if(!d_lastFrame->mvbOutlier[idx]) {

                Eigen::Vector3f x3Dw = pMP.mWorldPos;

                Eigen::Vector4f x3Dw_homogeneous;
                x3Dw_homogeneous.head<3>() = x3Dw;
                x3Dw_homogeneous[3] = 1.0f;

                Eigen::Vector4f x3Dc_homogeneous = transform_matrix * x3Dw_homogeneous;
                Eigen::Vector3f x3Dc = x3Dc_homogeneous.head<3>();

                const float xc = x3Dc(0);
                const float yc = x3Dc(1);
                const float invzc = 1.0/x3Dc(2);

                if(invzc<0) {
                    return;
                }

                Eigen::Vector2f uv;
                if (!cameraIsFisheye) {
                    uv = cameraProject_Pinhole(d_currentFrame->mpCamera_mvParameters, x3Dc);
                } else {
                    uv = cameraProject_KannalaBrandt8(d_currentFrame->mpCamera_mvParameters, x3Dc);
                }
            
                if(uv(0)<d_currentFrame->mnMinX || uv(0)>d_currentFrame->mnMaxX) {
                    return;
                }
                if(uv(1)<d_currentFrame->mnMinY || uv(1)>d_currentFrame->mnMaxY) {
                    return;
                }

                int nLastOctave = (d_lastFrame->Nleft == -1 || idx < d_lastFrame->Nleft) ? d_lastFrame->mvKeys[idx].octave
                                                                                    : d_lastFrame->mvKeysRight[idx - d_lastFrame->Nleft].octave;

                float x = uv(0);
                float y = uv(1);               
                float r = th * d_mvScaleFactors[nLastOctave];
                int minLevel, maxLevel;
                bool bRight = false;
                if(bForward) {
                    minLevel = nLastOctave;
                    maxLevel = -1;
                }
                else if(bBackward) {
                    minLevel = 0;
                    maxLevel = nLastOctave;        
                } else {
                    minLevel = nLastOctave-1;
                    maxLevel = nLastOctave+1;
                }
                uint8_t* MPdescriptor = &pMP.mDescriptor[0];

                int bestDist = 256;
                int bestIdx2 = -1;

                // ## GetFeaturesInArea Function ##
                float factorX = r;
                float factorY = r;

                const int nMinCellX = max(0, (int)floor((x - d_currentFrame->mnMinX - factorX) * d_currentFrame->mfGridElementWidthInv));
                if(nMinCellX >= FRAME_GRID_COLS) {
                    return;
                }

                const int nMaxCellX = min((int)FRAME_GRID_COLS-1, (int)ceil((x - d_currentFrame->mnMinX + factorX) * d_currentFrame->mfGridElementWidthInv));
                if(nMaxCellX < 0) {
                    return;
                }

                const int nMinCellY = max(0, (int)floor((y - d_currentFrame->mnMinY - factorY) * d_currentFrame->mfGridElementHeightInv));
                if(nMinCellY >= FRAME_GRID_ROWS) {
                    return;
                }

                const int nMaxCellY = min((int)FRAME_GRID_ROWS-1, (int)ceil((y - d_currentFrame->mnMinY + factorY) * d_currentFrame->mfGridElementHeightInv));
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
                            vCell = &d_currentFrame->flatMGrid[ix * FRAME_GRID_ROWS * KEYPOINTS_PER_CELL + iy * KEYPOINTS_PER_CELL];
                            vCell_size = d_currentFrame->flatMGrid_size[ix * FRAME_GRID_ROWS + iy];
                        } else {
                            vCell = &d_currentFrame->flatMGridRight[ix * FRAME_GRID_ROWS * KEYPOINTS_PER_CELL + iy * KEYPOINTS_PER_CELL];
                            vCell_size = d_currentFrame->flatMGridRight_size[ix * FRAME_GRID_ROWS + iy];
                        } 

                        if(vCell_size == 0) {                   
                            continue;
                        }

                        for(size_t j=0, jend=vCell_size; j<jend; j++)
                        {
                            const DATA_WRAPPER::CudaKeyPoint &kpUn = (d_currentFrame->Nleft == -1) ? d_currentFrame->mvKeysUn[vCell[j]]
                                                                    : (!bRight) ? d_currentFrame->mvKeys[vCell[j]]
                                                                                : d_currentFrame->mvKeysRight[vCell[j]];

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

                                if(!d_currentFrame->mvpMapPoints[vCell[j]].isEmpty)
                                    if(d_currentFrame->mvpMapPoints[vCell[j]].nObs > 0)
                                        continue;

                                if(d_currentFrame->Nleft == -1 && d_currentFrame->mvuRight[vCell[j]]>0)
                                {
                                    const float ur = uv(0) - d_currentFrame->mbf*invzc;
                                    const float er = fabs(ur - d_currentFrame->mvuRight[vCell[j]]);
                                    if(er>r)
                                        continue;
                                }

                                const uint8_t* d = &d_currentFrame->mDescriptors[vCell[j] * DESCRIPTOR_SIZE];     

                                const int dist = DescriptorDistance(MPdescriptor, d);

                                if(dist<bestDist)
                                {
                                    bestDist=dist;
                                    bestIdx2=vCell[j];
                                }
                            }
                        }
                    }
                }
                d_bestDist[idx] = bestDist;
                d_bestIdx2[idx] = bestIdx2;

                if(d_currentFrame->Nleft != -1){
                    Eigen::Vector4f x3Dc_homogeneous;
                    x3Dc_homogeneous.head<3>() = x3Dc;
                    x3Dc_homogeneous[3] = 1.0f;
                    Eigen::Vector4f x3Dr_homogeneous = d_currentFrame->mTrl * x3Dc_homogeneous;
                    Eigen::Vector3f x3Dr = x3Dr_homogeneous.head<3>();
                    uv = cameraProject_KannalaBrandt8(d_currentFrame->mpCamera_mvParameters, x3Dr);               

                    nLastOctave = (d_lastFrame->Nleft == -1 || idx < d_lastFrame->Nleft) ? d_lastFrame->mvKeys[idx].octave
                                                                                        : d_lastFrame->mvKeysRight[idx - d_lastFrame->Nleft].octave;
                                                                                        
                    x = uv(0);
                    y = uv(1);               
                    r = th * d_mvScaleFactors[nLastOctave];
                    bRight = true;
                    if(bForward) {
                        minLevel = nLastOctave;
                        maxLevel = -1;
                    }
                    else if(bBackward) {
                        minLevel = 0;
                        maxLevel = nLastOctave;        
                    } else {
                        minLevel = nLastOctave-1;
                        maxLevel = nLastOctave+1;
                    }
                    
                    MPdescriptor = &pMP.mDescriptor[0];

                    int bestDistR = 256;
                    int bestIdxR2 = -1;

                    // ## GetFeaturesInArea Function ##
                    float factorX = r;
                    float factorY = r;

                    const int nMinCellX = max(0, (int)floor((x - d_currentFrame->mnMinX - factorX) * d_currentFrame->mfGridElementWidthInv));
                    if(nMinCellX >= FRAME_GRID_COLS) {
                        return;
                    }

                    const int nMaxCellX = min((int)FRAME_GRID_COLS-1, (int)ceil((x - d_currentFrame->mnMinX + factorX) * d_currentFrame->mfGridElementWidthInv));
                    if(nMaxCellX < 0) {
                        return;
                    }

                    const int nMinCellY = max(0, (int)floor((y - d_currentFrame->mnMinY - factorY) * d_currentFrame->mfGridElementHeightInv));
                    if(nMinCellY >= FRAME_GRID_ROWS) {
                        return;
                    }

                    const int nMaxCellY = min((int)FRAME_GRID_ROWS-1, (int)ceil((y - d_currentFrame->mnMinY + factorY) * d_currentFrame->mfGridElementHeightInv));
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
                                vCell = &d_currentFrame->flatMGrid[ix * FRAME_GRID_ROWS * KEYPOINTS_PER_CELL + iy * KEYPOINTS_PER_CELL];
                                vCell_size = d_currentFrame->flatMGrid_size[ix * FRAME_GRID_ROWS + iy];
                            } else {
                                vCell = &d_currentFrame->flatMGridRight[ix * FRAME_GRID_ROWS * KEYPOINTS_PER_CELL + iy * KEYPOINTS_PER_CELL];
                                vCell_size = d_currentFrame->flatMGridRight_size[ix * FRAME_GRID_ROWS + iy];
                            } 

                            if(vCell_size == 0) {                   
                                continue;
                            }

                            for(size_t j=0, jend=vCell_size; j<jend; j++)
                            {
                                const DATA_WRAPPER::CudaKeyPoint &kpUn = (d_currentFrame->Nleft == -1) ? d_currentFrame->mvKeysUn[vCell[j]]
                                                                        : (!bRight) ? d_currentFrame->mvKeys[vCell[j]]
                                                                                    : d_currentFrame->mvKeysRight[vCell[j]];

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
                                    if(!d_currentFrame->mvpMapPoints[vCell[j] + d_currentFrame->Nleft].isEmpty)
                                        if(d_currentFrame->mvpMapPoints[vCell[j] + d_currentFrame->Nleft].nObs > 0)
                                            continue;

                                    const uint8_t* d = &d_currentFrame->mDescriptors[(vCell[j] + d_currentFrame->Nleft) * DESCRIPTOR_SIZE];
                        
                                    const int dist = DescriptorDistance(MPdescriptor, d);

                                    if(dist<bestDist)
                                    {
                                        bestDistR=dist;
                                        bestIdxR2=vCell[j];
                                    }
                                }
                            }
                        }
                    }
                    d_bestDistR[idx] = bestDistR;
                    d_bestIdxR2[idx] = bestIdxR2;
                }
            }
        }
    }
}

void PoseEstimationKernel::launch(ORB_SLAM3::Frame &CurrentFrame, const ORB_SLAM3::Frame &LastFrame,
                                const float th, const bool bForward, const bool bBackward, Eigen::Matrix4f transform_matrix,
                                int* h_bestDist, int* h_bestIdx2, int* h_bestDistR, int* h_bestIdxR2){

#ifdef REGISTER_STATS
    std::chrono::steady_clock::time_point startTotal = std::chrono::steady_clock::now();
#endif

    if (!memory_is_initialized){
        initialize();
    }

#ifdef REGISTER_STATS
    std::chrono::steady_clock::time_point startKernel = std::chrono::steady_clock::now();
#endif
    int blockSize = 256;
    int numBlocks = (LastFrame.N + blockSize - 1) / blockSize;
    searchByProjectionKernel<<<numBlocks, blockSize>>>(d_currentFrame, d_lastFrame,
                                                        CudaUtils::d_mvScaleFactors, CudaUtils::cameraIsFisheye,
                                                        th, bForward, bBackward, transform_matrix,
                                                        d_bestDist, d_bestIdx2, d_bestDistR, d_bestIdxR2);
    checkCudaError(cudaDeviceSynchronize(), "[PoseEstimationKernel:] Kernel launch failed"); 
#ifdef REGISTER_STATS
    std::chrono::steady_clock::time_point endKernel = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point startOutputTransfer = std::chrono::steady_clock::now();
#endif 

    checkCudaError(cudaMemcpy(h_bestDist, d_bestDist, LastFrame.N  * sizeof(int), cudaMemcpyDeviceToHost), "Failed to copy d_bestDist back to host");
    checkCudaError(cudaMemcpy(h_bestIdx2, d_bestIdx2, LastFrame.N  * sizeof(int), cudaMemcpyDeviceToHost), "Failed to copy d_bestIdx2 back to host");
    checkCudaError(cudaMemcpy(h_bestDistR, d_bestDistR, LastFrame.N  * sizeof(int), cudaMemcpyDeviceToHost), "Failed to copy d_bestDistR back to host");
    checkCudaError(cudaMemcpy(h_bestIdxR2, d_bestIdxR2, LastFrame.N  * sizeof(int), cudaMemcpyDeviceToHost), "Failed to copy d_bestIdxR2 back to host");

#ifdef REGISTER_STATS
    std::chrono::steady_clock::time_point endOutputTransfer = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point endTotal = std::chrono::steady_clock::now();

    double inputTransfer = input_data_transfer_time.back().second;
    double dataWrap = data_wrap_time.back().second;
    double outputTransfer = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(endOutputTransfer - startOutputTransfer).count();
    double kernel = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(endKernel - startKernel).count();
    double total = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(endTotal - startTotal).count();

    data_transfer_time.emplace_back(CurrentFrame.mnId, inputTransfer + outputTransfer);
    output_data_transfer_time.emplace_back(CurrentFrame.mnId, outputTransfer);
    
    kernel_exec_time.emplace_back(CurrentFrame.mnId, kernel);
    
    total_exec_time.emplace_back(CurrentFrame.mnId, total + inputTransfer + dataWrap);
#endif 

    return;
}

void PoseEstimationKernel::shutdown(){
    if(!memory_is_initialized)
        return;
    cudaFree(d_currentFrame);
    cudaFree(d_lastFrame);
    cudaFree(d_bestDist);
    cudaFree(d_bestIdx2);
    cudaFree(d_bestDistR);
    cudaFree(d_bestIdxR2);
}

void PoseEstimationKernel::saveStats(const string &file_path){

    string data_path = file_path + "/PoseEstimationKernel/";
    cout << "[PoseEstimationKernel:] writing stats data into file: " << data_path << '\n';
    if (mkdir(data_path.c_str(), 0755) == -1) {
        std::cerr << "[PoseEstimationKernel:] Error creating directory: " << strerror(errno) << std::endl;
    }

    ofstream myfile;

    myfile.open(data_path + "/total_exec_time.txt");
    for (const auto& p : total_exec_time) {
        myfile << p.first << ": " << p.second << std::endl;
    }
    myfile.close();

    myfile.open(data_path + "/data_transfer_time.txt");
    for (const auto& p : data_transfer_time) {
        myfile << p.first << ": " << p.second << std::endl;
    }
    myfile.close();

    myfile.open(data_path + "/data_wrap_time.txt");
    for (const auto& p : data_wrap_time) {
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
}