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
#include <stdio.h>

#include "resize.h"

#define index(x, y, step) (y * step + x)

__global__ void resize_kernel(uint old_h, uint old_w, float *_scaleFactor, const uchar *original_img, uchar *new_images, uint maxLevel, uint imageStep) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int level = (blockIdx.z * blockDim.z + threadIdx.z);

    if (level >= maxLevel){
        return;
    }
        

    const float scaleFactor = _scaleFactor[level];
    const uint new_h = round(old_h * 1/scaleFactor);
    const uint new_w = round(old_w * 1/scaleFactor);
    if (x >= new_w || y >= new_h){
        return;
    }
    
    uchar *new_image = &(new_images[level*old_h*old_w]);
    const uint newImageStep = new_w;
    uchar *newPixel = &(new_image[index(x, y, newImageStep)]);

    const float old_x = x * scaleFactor;
    const float old_y = y * scaleFactor;
    const int x_floor = floor(old_x);
    const int x_ceil = min(old_w - 1, (int)ceil(old_x));
    const int y_floor = floor(old_y);
    const int y_ceil = min(old_h - 1, (int)ceil(old_y));

    const uchar v1 = original_img[index(x_floor, y_floor, imageStep)];
    const uchar v2 = original_img[index(x_ceil, y_floor, imageStep)];
    const uchar v3 = original_img[index(x_floor, y_ceil, imageStep)];
    const uchar v4 = original_img[index(x_ceil, y_ceil, imageStep)];

    const float q1 = (x_ceil != x_floor) ? (v1 * ((x_ceil - old_x)/(x_ceil-x_floor)) + v2 * ((old_x - x_floor)/(x_ceil-x_floor))) : (x_ceil == x_floor) * v1;
    const float q2 = (x_ceil != x_floor) ? (v3 * ((x_ceil - old_x)/(x_ceil-x_floor)) + v4 * ((old_x - x_floor)/(x_ceil-x_floor))) : (x_ceil == x_floor) * v4;
    const float q = (y_ceil != y_floor) ? (q1 * ((y_ceil - old_y)/(y_ceil-y_floor)) + q2 * ((old_y - y_floor)/(y_ceil-y_floor))) : (y_ceil == y_floor) * q1;

    *newPixel = q;
}


void resize(uint old_h, uint old_w, float *_scaleFactor, uchar *original_img, uchar *new_images, uint maxLevel, uint imageStep, cudaStream_t stream) {
    dim3 dg( ceil( (float)old_w/128 ), ceil( (float)old_h/8 ), ceil( (float)maxLevel/1 ) );
    dim3 db( 128, 8, 1 );

    resize_kernel<<<dg, db, 0, stream>>>(old_h, old_w, _scaleFactor, original_img, new_images, maxLevel, imageStep);
}