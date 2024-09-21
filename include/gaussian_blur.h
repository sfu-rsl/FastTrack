/**
* This file is part of Cuda accelerated ORB-SLAM project by Filippo Muzzini, Nicola Capodieci, Roberto Cavicchioli and Benjamin Rouxel.
 * Implemented by Filippo Muzzini.
 *
 * Based on ORB-SLAM2 (Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós) and ORB-SLAM3 (Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós)
 *
 * Project under GPLv3 Licence
*
*/

#ifndef GAUSSIAN_BLUR
#define GAUSSIAN_BLUR

#include <opencv2/core/hal/interface.h>
#include <cuda.h>

void gaussian_blur( uchar *images, uchar *inputImage, uchar *imagesBlured, uchar *inputImageBlured, float *kernel, int cols, int rows, int inputImageStep, float* mvScaleFactor, int maxLevel, cudaStream_t cudaStream);

#endif