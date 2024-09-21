#ifndef STATS_H
#define STATS_H

#include <map>
#include <iostream>
#include <fstream>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include "Kernels/CudaUtils.h"
#include "Kernels/KernelController.h"

using namespace std;

class Stats {
    public:
        static void saveStats(const string &file_path);

    public:
        static std::vector<std::pair<long unsigned int, double>> tracking_time;
        static std::vector<std::pair<long unsigned int, double>> orbExtraction_time;
        static std::vector<std::pair<long unsigned int, double>> stereoMatch_time;
        static std::vector<std::pair<long unsigned int, double>> trackWithMotionModel_time;
        static std::vector<std::pair<long unsigned int, double>> TWM_poseEstimation_time;
        static std::vector<std::pair<long unsigned int, double>> TWM_poseOptimization_time;
        static std::vector<std::pair<long unsigned int, double>> relocalization_time;
        static std::vector<std::pair<long unsigned int, double>> trackLocalMap_time;
        static std::vector<std::pair<long unsigned int, double>> updateLocalMap_time;
        static std::vector<std::pair<long unsigned int, double>> updateLocalKF_time;
        static std::vector<std::pair<long unsigned int, double>> updateLocalPoints_time;
        static std::vector<std::pair<long unsigned int, double>> searchLocalPoints_time;
        static std::vector<std::pair<long unsigned int, double>> SLP_frameMapPointsItr_time;
        static std::vector<std::pair<long unsigned int, double>> SLP_localMapPointsItr_time;
        static std::vector<std::pair<long unsigned int, double>> SLP_searchByProjection_time;
        static std::vector<std::pair<long unsigned int, double>> TLM_poseOptimization_time;
        static std::vector<std::pair<long unsigned int, int>> num_local_mappoints;

        static double orbExtraction_init_time;
        static double stereoMatch_init_time;
        static double searchLocalPoints_init_time;
        static double poseEstimation_init_time;
        static int num_frames_lost;
};

#endif 