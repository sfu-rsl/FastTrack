/**
* This file is part of Cuda accelerated ORB-SLAM project by Filippo Muzzini, Nicola Capodieci, Roberto Cavicchioli and Benjamin Rouxel.
 * Implemented by Filippo Muzzini.
 *
 * Based on ORB-SLAM2 (Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós) and ORB-SLAM3 (Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós)
 *
 * Project under GPLv3 Licence
*
*/

#ifndef FAST_GPU_H
#define FAST_GPU_H

#include <opencv2/core/hal/interface.h>

#include "ORBextractor.h"

void fast_extract( uchar *images, uchar *inputImage, uint8_t th, uint8_t th_low, uint8_t *d_Rs, uint8_t *d_Rs_low, int *points, int n_, ORB_SLAM3::GpuPoint *buffers, uint *sizes, int cols, int rows, int inputImageStep, float* mvScaleFactor, int maxLevel, cudaStream_t cudaStream, cudaEvent_t interComplete, cv::Mat pyramid);

#endif