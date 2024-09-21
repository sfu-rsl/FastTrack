/**
* This file is part of Cuda accelerated ORB-SLAM project by Filippo Muzzini, Nicola Capodieci, Roberto Cavicchioli and Benjamin Rouxel.
 * Implemented by Filippo Muzzini.
 *
 * Based on ORB-SLAM2 (Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós) and ORB-SLAM3 (Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós)
 *
 * Project under GPLv3 Licence
*
*/

#ifndef DESCRIPTOR
#define DESCRIPTOR

#include "ORBextractor.h"

void compute_descriptor(uchar *images, uchar *inputImage, ORB_SLAM3::GpuPoint *points, uint *sizes, int maxPointsLevel, cv::Point* pattern, int inputImageStep, int maxLevel, int cols, int rows, float *mvScaleFactor, cudaStream_t cudaStream);

#endif