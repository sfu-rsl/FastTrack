/**
* This file is part of Cuda accelerated ORB-SLAM project by Filippo Muzzini, Nicola Capodieci, Roberto Cavicchioli and Benjamin Rouxel.
 * Implemented by Filippo Muzzini.
 *
 * Based on ORB-SLAM2 (Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós) and ORB-SLAM3 (Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós)
 *
 * Project under GPLv3 Licence
*
*/

#include <cuda.h>
#include <opencv2/core/hal/interface.h>

#include "gaussian_blur.h"
#include "ORBextractor.h"

__global__ void gaussian_blur_kernel(uint old_h, uint old_w, float *_scaleFactor, const uchar *original_img, const uchar *images, uchar *original_img_blurred, uchar *images_blurred, float *kernel, uint maxLevel, uint inputImageStep) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int level = blockIdx.z * blockDim.z + threadIdx.z;

    if (level >= maxLevel){
        return;
    }
        

    const float scaleFactor = _scaleFactor[level];
    const uint new_rows = round(old_h * 1/scaleFactor);
    const uint new_cols = round(old_w * 1/scaleFactor);
    if (x >= new_cols || y >= new_rows){
        return;
    }

    const int imageStep = (level == 0) * inputImageStep + (level != 0) * new_cols;
    const int image_index = x + y * imageStep;

    const uchar* im[2] = {original_img, &(images[(level*old_w*old_h)])};
    const int imIndex = (level != 0);

    const uchar *image = im[imIndex];

    uchar* imBlured[2] = {original_img_blurred, &(images_blurred[(level*old_w*old_h)])};
    uchar *imageBlured = imBlured[imIndex];


    float acc = 0;
    for (int w = -KW/2; w<=KW/2; w++)
        for (int h = -KH/2; h<=KH/2; h++) {
            const int index = min(max(image_index+(h*imageStep)+w, 0), new_cols*new_rows);
            acc += image[index] * kernel[(h + KH/2) * KW + (w + KW/2)];
        }
    
    imageBlured[image_index] = round(acc);
}

void gaussian_blur( uchar *images, uchar *inputImage, uchar *imagesBlured, uchar *inputImageBlured, float *kernel, int cols, int rows, int inputImageStep, float* mvScaleFactor, int maxLevel, cudaStream_t cudaStream) {
    dim3 dg( ceil( (float)cols/64 ), ceil( (float)rows/8 ), ceil( (float)maxLevel/1 ) );
    dim3 db( 64, 8, 1 );

    gaussian_blur_kernel<<<dg, db, 0, cudaStream>>>(rows, cols, mvScaleFactor, inputImage, images, inputImageBlured, imagesBlured, kernel, maxLevel, inputImageStep);
}