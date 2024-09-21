#include <iostream>
#include <cmath>

#include "Kernels/StereoMatchKernel.h"
#include <fstream>

__device__ double normL1(const uchar* src1, const uchar* src2, int imgWidth, int level, int origImageSize,
                         int startRow1, int endRow1, int startCol1, int endCol1,
                         int startRow2, int endRow2, int startCol2, int endCol2) {
    
    double l1Norm = 0.0;

    // Calculate patch width and height
    int patchWidth = endCol1 - startCol1;
    int patchHeight = endRow1 - startRow1;

    // Iterate through the specified patch area
    for (int row = 0; row < patchHeight; row++) {
        for (int col = 0; col < patchWidth; col++) {
            int idx1 = level * origImageSize + (startRow1 + row) * imgWidth + (startCol1 + col);
            int idx2 = level * origImageSize + (startRow2 + row) * imgWidth + (startCol2 + col);
            unsigned char val1 = src1[idx1];
            unsigned char val2 = src2[idx2];
            l1Norm += std::abs(static_cast<int>(val1) - static_cast<int>(val2));
        }
    }
    return l1Norm;
}

__global__ void stereoMatchKernel(const int N, int* vRowIndices, uchar* imagePyramidL, uchar* imagePyramidR, DATA_WRAPPER::CudaKeyPoint *keypointsL, 
                                  DATA_WRAPPER::CudaKeyPoint *keypointsR, uchar* descriptorsL, uchar* descriptorsR, int origRows, int origCols, 
                                  const int rowPadding, const int vRowIndicesCountInRow, const float scaleFactor, const float minD, const float maxD, 
                                  const int thOrbDist, const int minGlobalDist, const float mbf, int* vDistIdx, float* mvuRight, float* mvDepth) {

    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx >= N)
        return;

    mvDepth[idx]= -1.0f;
    mvuRight[idx] = -1.0f;
    vDistIdx[2*idx + 1] = -1;

    const DATA_WRAPPER::CudaKeyPoint &kpL = keypointsL[idx];
    const int &levelL = kpL.octave;
    const float &vL = kpL.pty;
    const float &uL = kpL.ptx;

    const float minU = uL - maxD;
    const float maxU = uL - minD;

    if (maxU < 0)
        return;

    int bestDist = minGlobalDist;
    int bestIdxR = -1;

    // Compare descriptor to right keypoints
    for (int iC = vRowIndicesCountInRow*(int)vL; iC < vRowIndicesCountInRow*((int)vL + 1); iC++) {
        const int iR = vRowIndices[iC];
        if (iR == -1)
            break;

        const DATA_WRAPPER::CudaKeyPoint &kpR = keypointsR[iR];

        if (kpR.octave < levelL-1 || kpR.octave > levelL+1)
            continue;

        const float &uR = kpR.ptx;

        if (uR >= minU && uR <= maxU) {
            const int dist = DescriptorDistance(&descriptorsL[idx*DESCRIPTOR_SIZE], &descriptorsR[iR*DESCRIPTOR_SIZE]);

            if (dist < bestDist) {
                bestDist = dist;
                bestIdxR = iR;
            }
        }
    }

    if (bestIdxR == -1)
        return;

    // Subpixel match by correlation
    if (bestDist < thOrbDist) {
        // coordinates in image pyramid at keypoint scale
        const float uR0 = keypointsR[bestIdxR].ptx;
        const float currScaleFactor = pow(scaleFactor, kpL.octave);
        const float currScaleFactorInv = 1.0f / currScaleFactor;
        const float scaleduL = round(kpL.ptx * currScaleFactorInv);
        const float scaledvL = round(kpL.pty * currScaleFactorInv);
        const float scaleduR0 = round(uR0 * currScaleFactorInv);

        // sliding window search
        const int w = 5;

        int bestDist = INT_MAX;
        int bestincR = 0;
        const int L = 5;
        float vDists[2*L+1];

        const float iniu = scaleduR0 + L - w;
        const float endu = scaleduR0 + L + w + 1;

        const int currLevelImgWidth = round((float)origCols * currScaleFactorInv);

        if (iniu < 0 || endu >= currLevelImgWidth)
            return;
        
        for(int incR = -L; incR <= L; incR++) {
            float dist = normL1(imagePyramidL, imagePyramidR, currLevelImgWidth, kpL.octave, origCols*origRows, scaledvL-w, scaledvL+w+1, 
                                scaleduL-w, scaleduL+w+1, scaledvL-w, scaledvL+w+1, scaleduR0+incR-w, scaleduR0+incR+w+1);
            if (dist < bestDist) {
                bestDist = dist;
                bestincR = incR;
            }
            vDists[L + incR] = dist;
        }

        if(bestincR == -L || bestincR == L)
            return;

        // Sub-pixel match (Parabola fitting)
        const float dist1 = vDists[L + bestincR - 1];
        const float dist2 = vDists[L + bestincR];
        const float dist3 = vDists[L + bestincR + 1];

        const float deltaR = (dist1-dist3) / (2.0f*(dist1+dist3-2.0f*dist2));

        if (deltaR < -1 || deltaR > 1)
            return;

        // Re-scaled coordinate
        float bestuR = currScaleFactor * ((float)scaleduR0 + (float)bestincR + deltaR);

        float disparity = uL - bestuR;

        if(disparity >= minD && disparity < maxD) {
            if (disparity <= 0) {
                disparity = 0.01;
                bestuR = uL-0.01;
            }
            mvDepth[idx] = mbf / disparity;
            mvuRight[idx] = bestuR;
            vDistIdx[2*idx] = bestDist;
            vDistIdx[2*idx + 1] = idx;
        }
    }
}

__global__ void findBestStereoMatchKernel(const int N, int* vRowIndices, DATA_WRAPPER::CudaKeyPoint *keypointsL, DATA_WRAPPER::CudaKeyPoint *keypointsR, 
                                          uchar* descriptorsL, uchar* descriptorsR, const int vRowIndicesCountInRow, const float minD, const float maxD, 
                                          const int thOrbDist, const int minGlobalDist, int *bestIdxR) {

    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx >= N)
        return;

    const DATA_WRAPPER::CudaKeyPoint &kpL = keypointsL[idx];
    const int &levelL = kpL.octave;
    const float &vL = kpL.pty;
    const float &uL = kpL.ptx;

    const float minU = uL - maxD;
    const float maxU = uL - minD;

    if (maxU < 0)
        return;

    int bestDist = minGlobalDist;
    bestIdxR[idx] = -1;

    // Compare descriptor to right keypoints
    for (int iC = vRowIndicesCountInRow*(int)vL; iC < vRowIndicesCountInRow*((int)vL + 1); iC++) {
        const int iR = vRowIndices[iC];
        if (iR == -1)
            break;

        const DATA_WRAPPER::CudaKeyPoint &kpR = keypointsR[iR];

        if (kpR.octave < levelL-1 || kpR.octave > levelL+1)
            continue;

        const float &uR = kpR.ptx;

        if (uR >= minU && uR <= maxU) {
            const int dist = DescriptorDistance(&descriptorsL[idx*DESCRIPTOR_SIZE], &descriptorsR[iR*DESCRIPTOR_SIZE]);

            if (dist < bestDist) {
                bestDist = dist;
                bestIdxR[idx] = iR;
            }
        }
    }

    if (bestDist >= thOrbDist)
        bestIdxR[idx] = -1;
}

__global__ void refineStereoMatchKernel(int *bestIdxR, uchar* imagePyramidL, uchar* imagePyramidR, DATA_WRAPPER::CudaKeyPoint *keypointsL, 
                                        DATA_WRAPPER::CudaKeyPoint *keypointsR, const int rowPadding, const int origRows, const int origCols, 
                                        const float scaleFactor, const float minD, const float maxD, const float mbf, 
                                        int* vDistIdx, float* mvuRight, float* mvDepth) {
    
    int threadIndexInBlock = threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;
    
    __shared__ DATA_WRAPPER::CudaKeyPoint kpL;
    __shared__ int levelL;
    __shared__ float uL, vL;
    __shared__ int currBestIdxR;
    __shared__ float uR0, currScaleFactor, currScaleFactorInv, scaleduL, scaledvL, scaleduR0;
    __shared__ float iniu, endu;
    __shared__ int currLevelImgWidth;
    __shared__ float l1Norm[2*SLIDING_WINDOW_SEARCH_SIZE_L + 1];

    if (threadIndexInBlock == 0) {
        mvDepth[blockIdx.x]= -1.0f;
        mvuRight[blockIdx.x] = -1.0f;
        vDistIdx[2*blockIdx.x] = -1;
        vDistIdx[2*blockIdx.x + 1] = -1;
    }

    if (bestIdxR[blockIdx.x] == -1)
        return;

    if (threadIndexInBlock == 0) {
        kpL = keypointsL[blockIdx.x];
        levelL = kpL.octave;
        uL = kpL.ptx;
        vL = kpL.pty;
        currBestIdxR = bestIdxR[blockIdx.x];
        uR0 = keypointsR[currBestIdxR].ptx;
        currScaleFactor = pow(scaleFactor, levelL);
        currScaleFactorInv = 1.0f / currScaleFactor;
        scaleduL = round(uL * currScaleFactorInv);
        scaledvL = round(vL * currScaleFactorInv);
        scaleduR0 = round(uR0 * currScaleFactorInv);
        iniu = scaleduR0 + SLIDING_WINDOW_SEARCH_SIZE_L - SLIDING_WINDOW_SIZE_W;
        endu = scaleduR0 + SLIDING_WINDOW_SEARCH_SIZE_L + SLIDING_WINDOW_SIZE_W + 1;
        currLevelImgWidth = round((float)origCols * currScaleFactorInv) + rowPadding;

        for (int i = 0; i < 2*SLIDING_WINDOW_SEARCH_SIZE_L+1; i++)
            l1Norm[i] = 0.0;
    }

    __syncthreads();

    if (iniu < 0 || endu >= currLevelImgWidth)
        return;

    // sliding window search
    int idxL = levelL * (origCols + rowPadding) * origRows + (scaledvL-SLIDING_WINDOW_SIZE_W + threadIdx.y) * currLevelImgWidth + 
               (scaleduL-SLIDING_WINDOW_SIZE_W + threadIdx.z);
    int idxR = levelL * (origCols + rowPadding) * origRows + (scaledvL-SLIDING_WINDOW_SIZE_W + threadIdx.y) * currLevelImgWidth + 
               (scaleduR0+threadIdx.x-SLIDING_WINDOW_SEARCH_SIZE_L-SLIDING_WINDOW_SIZE_W + threadIdx.z);

    unsigned char val1 = imagePyramidL[idxL];
    unsigned char val2 = imagePyramidR[idxR];
    float diff = std::abs(static_cast<int>(val1) - static_cast<int>(val2));

    atomicAdd(&l1Norm[threadIdx.x], diff);
            
    __syncthreads();

    if (threadIndexInBlock == 0) {
        
        int bestDist = INT_MAX;
        int bestincR = 0;

        for (int i = 0; i < 2*SLIDING_WINDOW_SEARCH_SIZE_L+1; i++) {
            if (l1Norm[i] < bestDist) {
                bestDist = l1Norm[i];
                bestincR = i;
            }
        }


        if (bestincR == 0 || bestincR == 2*SLIDING_WINDOW_SEARCH_SIZE_L)
            return;

        // Sub-pixel match (Parabola fitting)
        const float dist1 = l1Norm[bestincR - 1];
        const float dist2 = l1Norm[bestincR];
        const float dist3 = l1Norm[bestincR + 1];

        const float deltaR = (dist1-dist3) / (2.0f*(dist1+dist3-2.0f*dist2));

        if (deltaR < -1 || deltaR > 1)
            return;

        // Re-scaled coordinate
        float bestuR = currScaleFactor * ((float)scaleduR0 + (float)(bestincR - SLIDING_WINDOW_SEARCH_SIZE_L) + deltaR);

        float disparity = uL - bestuR;

        if (disparity >= minD && disparity < maxD) {
            if (disparity <= 0) {
                disparity = 0.01;
                bestuR = uL-0.01;
            }

            mvDepth[blockIdx.x] = mbf / disparity;
            mvuRight[blockIdx.x] = bestuR;
            vDistIdx[2*blockIdx.x] = bestDist;
            vDistIdx[2*blockIdx.x + 1] = blockIdx.x;
        }
    }
}

__global__ void fisheyeStereoMatchKernel(const int N, const int Nr, uchar* descriptors, int* matches) {

    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx >= N)
        return;

    matches[idx]= -1;

    int bestDist = INT_MAX;
    int bestIdxR = -1;

    int secondBestDist = INT_MAX;
    int secondBestIdxR = -1;

    // Compare left descriptor to right descriptor
    for (int i = 0; i < Nr; i++) {

        const int dist = DescriptorDistance(&descriptors[idx*DESCRIPTOR_SIZE], &descriptors[(N+i) * DESCRIPTOR_SIZE]);

        if (dist < bestDist) {
            secondBestDist = bestDist;
            secondBestIdxR = bestIdxR;
            bestDist = dist;
            bestIdxR = i;
        }
        else if (dist < secondBestDist) {
            secondBestDist = dist;
            secondBestIdxR = i;
        }
    }

    if (bestIdxR == -1 || secondBestIdxR == -1)
        return;

    if (bestDist < 0.7 * secondBestDist)
        matches[idx] = bestIdxR;
}

// For normal Stereo
void StereoMatchKernel::launch(std::vector<std::vector<int>> &vRowIndices, uchar* d_imagePyramidL, uchar* d_imagePyramidR, 
                               std::vector<cv::Mat> &mvImagePyramid, std::vector<cv::Mat> &mvImagePyramidRight,
                               std::vector<cv::KeyPoint> &mvKeys, std::vector<cv::KeyPoint> &mvKeysRight, 
                               cv::Mat mDescriptors, cv::Mat mDescriptorsRight, 
                               const float minD, const float maxD, const int thOrbDist, const float mbf, const bool _mvImagePyramidOnGpu,
                               std::vector<std::pair<int, int>> &vDistIdx, std::vector<float> &mvuRight, std::vector<float> &mvDepth) {

#ifdef REGISTER_STATS
    std::chrono::steady_clock::time_point startTotal = std::chrono::steady_clock::now();
#endif

    mvImagePyramidOnGpu = _mvImagePyramidOnGpu;
    const int nLevels = CudaUtils::nLevels;
    const int nRows = CudaUtils::nRows;
    const int nCols = CudaUtils::nCols;
    const int minGlobalDist = CudaUtils::ORBmatcher_TH_HIGH;
    const float scaleFactor = CudaUtils::scaleFactor;
    const int N = mvKeys.size(), Nr = mvKeysRight.size();

    if (!memory_is_initialized)
        initialize();

#ifdef REGISTER_STATS
    std::chrono::steady_clock::time_point startCopyObjectCreation = std::chrono::steady_clock::now();
#endif

    int vRowIndicesFlat[nRows * MAX_FEATURES_IN_ROW_SLIDING_WINDOW] = {-1};
    flattenVRowIndices(vRowIndices, vRowIndicesFlat);

    DATA_WRAPPER::CudaKeyPoint gpuKeypointsL[N], gpuKeypointsR[Nr];
    copyGPUKeypoints(mvKeys, gpuKeypointsL);
    copyGPUKeypoints(mvKeysRight, gpuKeypointsR);

    int origImageSize = nRows * mvImagePyramid[0].step[0];
    uchar imagePyramidLFlat[nLevels * origImageSize], imagePyramidRFlat[nLevels*origImageSize];
    if (!mvImagePyramidOnGpu) {
        flattenPyramid(mvImagePyramid, origImageSize, imagePyramidLFlat);
        flattenPyramid(mvImagePyramidRight, origImageSize, imagePyramidRFlat);
    }

#ifdef REGISTER_STATS
    std::chrono::steady_clock::time_point endCopyObjectCreation = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point startMemcpy = std::chrono::steady_clock::now();
#endif

    checkCudaError(cudaMemcpy(d_rowIndices, vRowIndicesFlat, sizeof(int)*nRows*MAX_FEATURES_IN_ROW_SLIDING_WINDOW, cudaMemcpyHostToDevice), "Failed to copy vector vRowIndicesFlat from host to device");
    checkCudaError(cudaMemcpy(d_gpuKeypointsL, gpuKeypointsL, sizeof(DATA_WRAPPER::CudaKeyPoint)*N, cudaMemcpyHostToDevice), "Failed to copy vector gpuKeypointsL from host to device");
    checkCudaError(cudaMemcpy(d_gpuKeypointsR, gpuKeypointsR, sizeof(DATA_WRAPPER::CudaKeyPoint)*Nr, cudaMemcpyHostToDevice), "Failed to copy vector d_gpuKeypointsR from host to device");
    checkCudaError(cudaMemcpy(d_descriptorsL, mDescriptors.data, sizeof(uchar)*N*DESCRIPTOR_SIZE, cudaMemcpyHostToDevice), "Failed to copy vector mDescriptors from host to device");
    checkCudaError(cudaMemcpy(d_descriptorsR, mDescriptorsRight.data, sizeof(uchar)*Nr*DESCRIPTOR_SIZE, cudaMemcpyHostToDevice), "Failed to copy vector mDescriptorsRight from host to device");

    if (!mvImagePyramidOnGpu) {
        checkCudaError(cudaMemcpy(d_imagePyramidLCopied, imagePyramidLFlat, sizeof(uchar)*nLevels*origImageSize, cudaMemcpyHostToDevice), "Failed to copy vector d_imagePyramidLCopied from host to device");
        checkCudaError(cudaMemcpy(d_imagePyramidRCopied, imagePyramidRFlat, sizeof(uchar)*nLevels*origImageSize, cudaMemcpyHostToDevice), "Failed to copy vector d_imagePyramidRCopied from host to device");
    }

#ifdef REGISTER_STATS
    std::chrono::steady_clock::time_point endMemcpy = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point startStereoMatchKernel = std::chrono::steady_clock::now();
#endif
    int stereoMatchKernelBlockSize = 128;
    int stereoMatchKernelGridSize = (N + stereoMatchKernelBlockSize - 1) / stereoMatchKernelBlockSize;         

    findBestStereoMatchKernel<<<stereoMatchKernelGridSize, stereoMatchKernelBlockSize>>>(
        N, d_rowIndices, d_gpuKeypointsL, d_gpuKeypointsR, d_descriptorsL, d_descriptorsR, MAX_FEATURES_IN_ROW_SLIDING_WINDOW, 
        minD, maxD, thOrbDist, minGlobalDist, d_bestIdxR
    );

    checkCudaError(cudaGetLastError(), "Failed to launch findBestStereoMatchKernel kernel");
    checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize returned error code after launching the kernel");

#ifdef REGISTER_STATS
    std::chrono::steady_clock::time_point endStereoMatchKernel = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point startRefineKernel = std::chrono::steady_clock::now();
#endif

    dim3 refineMatchKernelBlockSize(2*SLIDING_WINDOW_SEARCH_SIZE_L+1, 2*SLIDING_WINDOW_SIZE_W+1, 2*SLIDING_WINDOW_SIZE_W+1);
    int refineMatchKernelGridSize = N;

    if (mvImagePyramidOnGpu) {
        refineStereoMatchKernel<<<refineMatchKernelGridSize, refineMatchKernelBlockSize>>>(
            d_bestIdxR, d_imagePyramidL, d_imagePyramidR, d_gpuKeypointsL, d_gpuKeypointsR, 0, nRows, nCols, scaleFactor, minD, maxD, mbf,
            d_vDistIdx, d_mvuRight, d_mvDepth
        );
    }
    else {
        refineStereoMatchKernel<<<refineMatchKernelGridSize, refineMatchKernelBlockSize>>>(
            d_bestIdxR, d_imagePyramidLCopied, d_imagePyramidRCopied, d_gpuKeypointsL, d_gpuKeypointsR, EDGE_THRESHOLD*2, nRows, nCols, scaleFactor, minD, maxD, mbf,
            d_vDistIdx, d_mvuRight, d_mvDepth
        );
    }

    checkCudaError(cudaGetLastError(), "Failed to launch findBestStereoMatchKernel kernel");
    checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize returned error code after launching the kernel");

#ifdef REGISTER_STATS
    std::chrono::steady_clock::time_point endRefineKernel = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point startMemcpyToCPU = std::chrono::steady_clock::now();
#endif

    int h_vDistIdx[2*N];
    float h_mvuRight[N], h_mvDepth[N];
    checkCudaError(cudaMemcpy(h_vDistIdx, d_vDistIdx, sizeof(int)*2*N, cudaMemcpyDeviceToHost), "Failed to copy vector d_vDistIdx from device to host");
    checkCudaError(cudaMemcpy(h_mvuRight, d_mvuRight, sizeof(float)*N, cudaMemcpyDeviceToHost), "Failed to copy vector d_mvuRight from device to host");
    checkCudaError(cudaMemcpy(h_mvDepth, d_mvDepth, sizeof(float)*N, cudaMemcpyDeviceToHost), "Failed to copy vector d_mvDepth from device to host");

#ifdef REGISTER_STATS
    std::chrono::steady_clock::time_point endMemcpyToCPU = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point startPostProcess = std::chrono::steady_clock::now();
#endif

    vDistIdx = convertToVectorOfPairs(h_vDistIdx, N);
    mvuRight = std::vector<float>(h_mvuRight, h_mvuRight + N);
    mvDepth = std::vector<float>(h_mvDepth, h_mvDepth + N);

#ifdef REGISTER_STATS
    std::chrono::steady_clock::time_point endPostProcess = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point endTotal = std::chrono::steady_clock::now();

    double copyObjectCreation = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(endCopyObjectCreation - startCopyObjectCreation).count();
    double memcpyToGPU = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(endMemcpy - startMemcpy).count();
    double stereoMatchKernel = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(endStereoMatchKernel - startStereoMatchKernel).count();
    double refineKernel = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(endRefineKernel - startRefineKernel).count();
    double memcpyToCPU = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(endMemcpyToCPU - startMemcpyToCPU).count();
    double postProcess = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(endPostProcess - startPostProcess).count();
    double total = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(endTotal - startTotal).count();

    data_wrap_time.emplace_back(frameCounter, copyObjectCreation + postProcess);
    input_data_wrap_time.emplace_back(frameCounter, copyObjectCreation);
    input_data_transfer_time.emplace_back(frameCounter, memcpyToGPU);
    kernel1_exec_time.emplace_back(frameCounter, stereoMatchKernel);
    kernel2_exec_time.emplace_back(frameCounter, refineKernel);
    kernel_exec_time.emplace_back(frameCounter, stereoMatchKernel + refineKernel);
    output_data_transfer_time.emplace_back(frameCounter, memcpyToCPU);
    data_transfer_time.emplace_back(frameCounter, memcpyToGPU + memcpyToCPU);
    output_data_wrap_time.emplace_back(frameCounter, postProcess);
    total_exec_time.emplace_back(frameCounter, total);
#endif

    frameCounter++;
}

// For fisheye
void StereoMatchKernel::launch(const int N, const int Nr, cv::Mat mDescriptors, cv::Mat mDescriptorsRight, int *matches) {

#ifdef REGISTER_STATS
    std::chrono::steady_clock::time_point startTotal = std::chrono::steady_clock::now();
#endif

    if (!memory_is_initialized)
        initialize();

#ifdef REGISTER_STATS
    std::chrono::steady_clock::time_point startMemcpy = std::chrono::steady_clock::now();
#endif

    checkCudaError(cudaMemcpy(d_descriptorsAll, mDescriptors.data, sizeof(uchar)*N*DESCRIPTOR_SIZE, cudaMemcpyHostToDevice), "Failed to copy vector mDescriptors from host to device");
    checkCudaError(cudaMemcpy(d_descriptorsAll + N*DESCRIPTOR_SIZE, mDescriptorsRight.data, sizeof(uchar)*Nr*DESCRIPTOR_SIZE, cudaMemcpyHostToDevice), "Failed to copy vector mDescriptorsRight from host to device");

#ifdef REGISTER_STATS
    std::chrono::steady_clock::time_point endMemcpy = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point startKernel = std::chrono::steady_clock::now();
#endif

    int blockSize = 128;
    int gridSize = (N + blockSize - 1) / blockSize;

    fisheyeStereoMatchKernel<<<gridSize, blockSize>>>(N, Nr, d_descriptorsAll, d_matches); 

    checkCudaError(cudaGetLastError(), "Failed to launch fisheye stereo match kernel");
    checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize returned error code after launching the kernel");

#ifdef REGISTER_STATS
    std::chrono::steady_clock::time_point endKernel = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point startMemcpyToCPU = std::chrono::steady_clock::now();
#endif

    checkCudaError(cudaMemcpy(matches, d_matches, sizeof(int)*N, cudaMemcpyDeviceToHost), "Failed to copy vector d_matches from device to host");

#ifdef REGISTER_STATS
    std::chrono::steady_clock::time_point endMemcpyToCPU = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point endTotal = std::chrono::steady_clock::now();

    double memcpyToGPU = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(endMemcpy - startMemcpy).count();
    double kernel = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(endKernel - startKernel).count();
    double memcpyToCPU = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(endMemcpyToCPU - startMemcpyToCPU).count();
    double total = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(endTotal - startTotal).count();

    input_data_transfer_time.emplace_back(frameCounter, memcpyToGPU);
    kernel_exec_time.emplace_back(frameCounter, kernel);
    output_data_transfer_time.emplace_back(frameCounter, memcpyToCPU);
    data_transfer_time.emplace_back(frameCounter, memcpyToGPU + memcpyToCPU);
    total_exec_time.emplace_back(frameCounter, total);
#endif

    frameCounter++;
}

void StereoMatchKernel::initialize() {
    
    if (memory_is_initialized)
        return;

    int maxFeatures = CudaUtils::nFeatures_with_th;
    int nLevels = CudaUtils::nLevels;
    int nRows = CudaUtils::nRows;
    int nCols = CudaUtils::nCols;

    if (CudaUtils::cameraIsFisheye) {
        checkCudaError(cudaMalloc((void**)&d_descriptorsAll, 2*sizeof(uchar)*maxFeatures*DESCRIPTOR_SIZE), "Failed to allocate device vector d_descriptorsAll");
        checkCudaError(cudaMalloc((void**)&d_matches, sizeof(int)*maxFeatures), "Failed to allocate device vector d_matches");
    }
    else {
        checkCudaError(cudaMalloc((void**)&d_rowIndices, sizeof(int)*nRows*MAX_FEATURES_IN_ROW_SLIDING_WINDOW), "Failed to allocate device vector d_rowIndices");
        checkCudaError(cudaMalloc((void**)&d_gpuKeypointsL, sizeof(DATA_WRAPPER::CudaKeyPoint)*maxFeatures), "Failed to allocate device vector d_gpuKeypointsL");
        checkCudaError(cudaMalloc((void**)&d_gpuKeypointsR, sizeof(DATA_WRAPPER::CudaKeyPoint)*maxFeatures), "Failed to allocate device vector d_gpuKeypointsR");
        checkCudaError(cudaMalloc((void**)&d_descriptorsL, sizeof(uchar)*maxFeatures*DESCRIPTOR_SIZE), "Failed to allocate device vector d_descriptorsL");
        checkCudaError(cudaMalloc((void**)&d_descriptorsR, sizeof(uchar)*maxFeatures*DESCRIPTOR_SIZE), "Failed to allocate device vector d_descriptorsR");
        checkCudaError(cudaMalloc((void**)&d_vDistIdx, sizeof(int)*2*maxFeatures), "Failed to allocate device vector d_vDistIdx");
        checkCudaError(cudaMalloc((void**)&d_mvuRight, sizeof(float)*maxFeatures), "Failed to allocate device vector d_mvuRight");
        checkCudaError(cudaMalloc((void**)&d_mvDepth, sizeof(float)*maxFeatures), "Failed to allocate device vector d_mvDepth");
        checkCudaError(cudaMalloc((void**)&d_bestIdxR, sizeof(int)*maxFeatures), "Failed to allocate device vector d_bestIdxR");

        if (!mvImagePyramidOnGpu) {
            int origImageSize = nLevels * nRows * (nCols + 2*EDGE_THRESHOLD);
            checkCudaError(cudaMalloc((void**)&d_imagePyramidLCopied, sizeof(uchar)*origImageSize), "Failed to allocate device vector d_imagePyramidLCopied");
            checkCudaError(cudaMalloc((void**)&d_imagePyramidRCopied, sizeof(uchar)*origImageSize), "Failed to allocate device vector d_imagePyramidRCopied");
        }
    }

    memory_is_initialized = true;
}

void StereoMatchKernel::shutdown() {
    if (memory_is_initialized) {
        if (CudaUtils::cameraIsFisheye) {
            cudaFree(d_descriptorsAll);
            cudaFree(d_matches);
        }
        else {
            cudaFree(d_rowIndices);
            cudaFree(d_gpuKeypointsL);
            cudaFree(d_gpuKeypointsR);
            cudaFree(d_descriptorsL);
            cudaFree(d_descriptorsR);
            cudaFree(d_vDistIdx);
            cudaFree(d_mvuRight);
            cudaFree(d_mvDepth);
            cudaFree(d_bestIdxR);

            if (mvImagePyramidOnGpu) {
                cudaFree(d_imagePyramidLCopied);
                cudaFree(d_imagePyramidRCopied);
            }
        }
    }
}

std::vector<std::pair<int, int>> StereoMatchKernel::convertToVectorOfPairs(int* X, int N) {
    std::vector<std::pair<int, int>> vec;
    for (int i = 0; i < 2 * N; i += 2) 
        if (X[i + 1] != -1)
            vec.push_back(std::make_pair(X[i], X[i + 1]));
    return vec;
}

void StereoMatchKernel::flattenVRowIndices(const std::vector<std::vector<int>>& input, int* flat) {
    for (int i = 0; i < input.size(); i++)
        memcpy(flat + i*MAX_FEATURES_IN_ROW_SLIDING_WINDOW, input[i].data(), sizeof(int) * input[i].size());
}

void StereoMatchKernel::copyGPUKeypoints(const std::vector<cv::KeyPoint> keypoints, DATA_WRAPPER::CudaKeyPoint* out) {
    for (int i = 0; i < keypoints.size(); i++) {
        // out[i] = DATA_WRAPPER::CudaKeyPoint(keypoints[i].octave, keypoints[i].pt.x, keypoints[i].pt.y);
        out[i].ptx = keypoints[i].pt.x;
        out[i].pty = keypoints[i].pt.y;
        out[i].octave = keypoints[i].octave;
    }
}

void StereoMatchKernel::flattenPyramid(std::vector<cv::Mat>& mvImagePyramid, int origImageSize, uchar* flat) {
    for (int i = 0; i < mvImagePyramid.size(); i++)
        memcpy(flat + i*origImageSize, mvImagePyramid[i].data, sizeof(uchar) * mvImagePyramid[i].rows * mvImagePyramid[i].step[0]);
}

void StereoMatchKernel::saveStats(const std::string &file_path){

    std::string data_path = file_path + "/StereoMatchKernel/";
    std::cout << "[StereoMatchKernel:] writing stats data into file: " << data_path << '\n';
    if (mkdir(data_path.c_str(), 0755) == -1) {
        std::cerr << "[StereoMatchKernel:] Error creating directory: " << strerror(errno) << std::endl;
    }
    std::ofstream myfile;
    
    myfile.open(data_path + "/total_exec_time.txt");
    for (const auto& p : total_exec_time) {
        myfile << p.first << ": " << p.second << std::endl;
    }
    myfile.close();

    myfile.open(data_path + "/kernel_exec_time.txt");
    for (const auto& p : kernel_exec_time) {
        myfile << p.first << ": " << p.second << std::endl;
    }
    myfile.close();

    myfile.open(data_path + "/data_transfer_time.txt");
    for (const auto& p : data_transfer_time) {
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

    if (!CudaUtils::cameraIsFisheye) {
        myfile.open(data_path + "/kernel1_exec_time.txt");
        for (const auto& p : kernel1_exec_time) {
            myfile << p.first << ": " << p.second << std::endl;
        }
        myfile.close();

        myfile.open(data_path + "/kernel2_exec_time.txt");
        for (const auto& p : kernel2_exec_time) {
            myfile << p.first << ": " << p.second << std::endl;
        }
        myfile.close();

        myfile.open(data_path + "/data_wrap_time.txt");
        for (const auto& p : data_wrap_time) {
            myfile << p.first << ": " << p.second << std::endl;
        }
        myfile.close();

        myfile.open(data_path + "/input_data_wrap_time.txt");
        for (const auto& p : input_data_wrap_time) {
            myfile << p.first << ": " << p.second << std::endl;
        }
        myfile.close();

        myfile.open(data_path + "/output_data_wrap_time.txt");
        for (const auto& p : output_data_wrap_time) {
            myfile << p.first << ": " << p.second << std::endl;
        }
        myfile.close();
    }
}