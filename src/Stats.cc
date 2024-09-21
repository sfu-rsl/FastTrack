#include "Stats.h"
#include <sstream>  

using namespace std;

std::vector<std::pair<long unsigned int, double>> Stats::tracking_time;
std::vector<std::pair<long unsigned int, double>> Stats::orbExtraction_time;
std::vector<std::pair<long unsigned int, double>> Stats::stereoMatch_time;
std::vector<std::pair<long unsigned int, double>> Stats::trackWithMotionModel_time;
std::vector<std::pair<long unsigned int, double>> Stats::TWM_poseEstimation_time;
std::vector<std::pair<long unsigned int, double>> Stats::TWM_poseOptimization_time;
std::vector<std::pair<long unsigned int, double>> Stats::relocalization_time;
std::vector<std::pair<long unsigned int, double>> Stats::trackLocalMap_time;
std::vector<std::pair<long unsigned int, double>> Stats::updateLocalMap_time;
std::vector<std::pair<long unsigned int, double>> Stats::updateLocalKF_time;
std::vector<std::pair<long unsigned int, double>> Stats::updateLocalPoints_time;
std::vector<std::pair<long unsigned int, double>> Stats::searchLocalPoints_time;
std::vector<std::pair<long unsigned int, double>> Stats::SLP_frameMapPointsItr_time;
std::vector<std::pair<long unsigned int, double>> Stats::SLP_localMapPointsItr_time;
std::vector<std::pair<long unsigned int, double>> Stats::SLP_searchByProjection_time;
std::vector<std::pair<long unsigned int, double>> Stats::TLM_poseOptimization_time;
std::vector<std::pair<long unsigned int, int>> Stats::num_local_mappoints;
double Stats::orbExtraction_init_time = 0;
double Stats::stereoMatch_init_time = 0;
double Stats::searchLocalPoints_init_time = 0;
double Stats::poseEstimation_init_time = 0;
int Stats::num_frames_lost = 0;

void Stats::saveStats(const string &file_path) {
#ifdef REGISTER_STATS
    string data_path = file_path + "/data/";
    cout << "Writing stats data into file: " << data_path << '\n';
    if (mkdir(data_path.c_str(), 0755) == -1) {
        std::cerr << "[Stats:] Error creating directory: " << strerror(errno) << std::endl;
    }

    KernelController::saveKernelsStats(data_path);

    std::ofstream myfile;

    myfile.open(data_path + "/tracking_time.txt");
    for (const auto& p : tracking_time) {
        myfile << p.first << ": " << p.second << std::endl;
    }
    myfile.close();

    myfile.open(data_path + "/orbExtraction_time.txt");
    for (const auto& p : orbExtraction_time) {
        myfile << p.first << ": " << p.second << std::endl;
    }
    myfile.close();

    myfile.open(data_path + "/stereoMatch_time.txt");
    for (const auto& p : stereoMatch_time) {
        myfile << p.first << ": " << p.second << std::endl;
    }
    myfile.close();

    myfile.open(data_path + "/trackWithMotionModel_time.txt");
    for (const auto& p : trackWithMotionModel_time) {
        myfile << p.first << ": " << p.second << std::endl;
    }
    myfile.close();

    myfile.open(data_path + "/TWM_poseEstimation_time.txt");
    for (const auto& p : TWM_poseEstimation_time) {
        myfile << p.first << ": " << p.second << std::endl;
    }
    myfile.close();

    myfile.open(data_path + "/TWM_poseOptimization_time.txt");
    for (const auto& p : TWM_poseOptimization_time) {
        myfile << p.first << ": " << p.second << std::endl;
    }
    myfile.close();

    myfile.open(data_path + "/relocalization_time.txt");
    for (const auto& p : relocalization_time) {
        myfile << p.first << ": " << p.second << std::endl;
    }
    myfile.close();

    myfile.open(data_path + "/trackLocalMap_time.txt");
    for (const auto& p : trackLocalMap_time) {
        myfile << p.first << ": " << p.second << std::endl;
    }
    myfile.close();

    myfile.open(data_path + "/updateLocalMap_time.txt");
    for (const auto& p : updateLocalMap_time) {
        myfile << p.first << ": " << p.second << std::endl;
    }
    myfile.close();

    myfile.open(data_path + "/updateLocalKF_time.txt");
    for (const auto& p : updateLocalKF_time) {
        myfile << p.first << ": " << p.second << std::endl;
    }
    myfile.close();

    myfile.open(data_path + "/updateLocalPoints_time.txt");
    for (const auto& p : updateLocalPoints_time) {
        myfile << p.first << ": " << p.second << std::endl;
    }
    myfile.close();

    myfile.open(data_path + "/searchLocalPoints_time.txt");
    for (const auto& p : searchLocalPoints_time) {
        myfile << p.first << ": " << p.second << std::endl;
    }
    myfile.close();

    myfile.open(data_path + "/SLP_frameMapPointsItr_time.txt");
    for (const auto& p : SLP_frameMapPointsItr_time) {
        myfile << p.first << ": " << p.second << std::endl;
    }
    myfile.close();

    myfile.open(data_path + "/SLP_localMapPointsItr_time.txt");
    for (const auto& p : SLP_localMapPointsItr_time) {
        myfile << p.first << ": " << p.second << std::endl;
    }
    myfile.close();

    myfile.open(data_path + "/SLP_searchByProjection_time.txt");
    for (const auto& p : SLP_searchByProjection_time) {
        myfile << p.first << ": " << p.second << std::endl;
    }
    myfile.close();

    myfile.open(data_path + "/TLM_poseOptimization_time.txt");
    for (const auto& p : TLM_poseOptimization_time) {
        myfile << p.first << ": " << p.second << std::endl;
    }
    myfile.close();

    myfile.open(data_path + "/num_local_mappoints.txt");
    for (const auto& p : num_local_mappoints) {
        myfile << p.first << ": " << p.second << std::endl;
    }
    myfile.close();

    myfile.open(data_path + "/kernel_initialization_time.txt");
    myfile << "total_initialization_time: " << orbExtraction_init_time + stereoMatch_init_time + searchLocalPoints_init_time + poseEstimation_init_time << std::endl;
    myfile << "orbExtraction_initialization_time: " << orbExtraction_init_time << std::endl;
    myfile << "stereoMatch_initialization_time: " << stereoMatch_init_time << std::endl;
    myfile << "searchLocalPoints_initialization_time: " << searchLocalPoints_init_time << std::endl;
    myfile << "poseEstimation_initialization_time: " << poseEstimation_init_time << std::endl;
    myfile.close();

    myfile.open(data_path + "/num_frames_lost.txt");
    myfile << num_frames_lost << std::endl;
    myfile.close(); 
#endif
}