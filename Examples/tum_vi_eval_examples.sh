#!/bin/bash
pathDatasetTUM_VI=$HOME/SLAM/Datasets/tumvi #Example, it is necesary to change it by the dataset path

# "optimized" or "original"
orbExtractionRunstatus=$1
stereoMatchRunstatus=$2
searchLocalPointsRunstatus=$3
poseEstimationRunstatus=$4
poseOptimizationRunstatus=$5
dataset_name=$6
version=$7

if [ "$poseOptimizationRunstatus" -eq 1 ]; then
    statsDir="../Results/poseOptimization_on/${orbExtractionRunstatus}${stereoMatchRunstatus}${searchLocalPointsRunstatus}${poseEstimationRunstatus}/${dataset_name}/${version}"
else
    statsDir="../Results/poseOptimization_off/${orbExtractionRunstatus}${stereoMatchRunstatus}${searchLocalPointsRunstatus}${poseEstimationRunstatus}/${dataset_name}/${version}"
fi

if [ ! -d "$statsDir" ]; then
    mkdir -p "$statsDir"
fi

# Single Session Example

# echo "Launching magistrale with Stereo-Inertial sensor"
file_name="dataset-${dataset_name}_stereoi"

# EXECUTABLE=./Stereo-Inertial/stereo_inertial_tum_vi
# ARGS="../Vocabulary/ORBvoc.txt Stereo-Inertial/TUM-VI.yaml ${pathDatasetTUM_VI}/dataset-${dataset_name}_512_16/mav0/cam0/data ${pathDatasetTUM_VI}/dataset-${dataset_name}_512_16/mav0/cam1/data Stereo-Inertial/TUM_TimeStamps/dataset-${dataset_name}_512.txt Stereo-Inertial/TUM_IMU/dataset-${dataset_name}_512.txt  ${file_name} ${statsDir} ${orbExtractionRunstatus} ${stereoMatchRunstatus} ${searchLocalPointsRunstatus} ${poseEstimationRunstatus}"
# gdb -ex "set args $ARGS" -ex "run" ./Stereo-Inertial/stereo_inertial_tum_vi
# compute-sanitizer --tool memcheck --report-api-errors all --show-backtrace no ./Stereo-Inertial/stereo_inertial_tum_vi ../Vocabulary/ORBvoc.txt Stereo-Inertial/TUM-VI.yaml ${pathDatasetTUM_VI}/dataset-${dataset_name}_512_16/mav0/cam0/data ${pathDatasetTUM_VI}/dataset-${dataset_name}_512_16/mav0/cam1/data Stereo-Inertial/TUM_TimeStamps/dataset-${dataset_name}_512.txt Stereo-Inertial/TUM_IMU/dataset-${dataset_name}_512.txt  ${file_name} ${optimization_status} ${statsDir}
./Stereo-Inertial/stereo_inertial_tum_vi ../Vocabulary/ORBvoc.txt Stereo-Inertial/TUM-VI.yaml ${pathDatasetTUM_VI}/dataset-${dataset_name}_512_16/mav0/cam0/data ${pathDatasetTUM_VI}/dataset-${dataset_name}_512_16/mav0/cam1/data Stereo-Inertial/TUM_TimeStamps/dataset-${dataset_name}_512.txt Stereo-Inertial/TUM_IMU/dataset-${dataset_name}_512.txt  ${file_name} ${statsDir} ${orbExtractionRunstatus} ${stereoMatchRunstatus} ${searchLocalPointsRunstatus} ${poseEstimationRunstatus} ${poseOptimizationRunstatus}
# ./Stereo/stereo_tum_vi ../Vocabulary/ORBvoc.txt Stereo-Inertial/TUM-VI.yaml ${pathDatasetTUM_VI}/dataset-${dataset_name}_512_16/mav0/cam0/data ${pathDatasetTUM_VI}/dataset-${dataset_name}_512_16/mav0/cam1/data Stereo-Inertial/TUM_TimeStamps/dataset-${dataset_name}_512.txt ${file_name} ${statsDir} ${orbExtractionRunstatus} ${stereoMatchRunstatus} ${searchLocalPointsRunstatus} ${poseEstimationRunstatus}

# echo "------------------------------------"

echo "Evaluation of ${dataset_name} trajectory with Stereo-Inertial sensor"
python3 -W ignore ../evaluation/evaluate3.py "$pathDatasetTUM_VI"/dataset-${dataset_name}_512_16//mav0/mocap0/data.csv f_${file_name}.txt --plot ${dataset_name}_512_stereoi.pdf --verbose
echo "Plotting data"
python3 ../plot.py "${statsDir}"
python3 ../plot_piechart.py "${statsDir}"

files=("f_dataset-${dataset_name}_stereoi.csv"
"f_dataset-${dataset_name}_stereoi.txt"
"f_dataset-${dataset_name}_stereoi.png"
"kf_dataset-${dataset_name}_stereoi.txt"
)
destination_directory="${statsDir}/trajectory"
mkdir -p $destination_directory
mv "${files[@]}" "$destination_directory"


#Multi Session Example
# ./Stereo-Inertial/stereo_inertial_tum_vi ../Vocabulary/ORBvoc.txt Stereo-Inertial/TUM-VI.yaml "$pathDatasetTUM_VI"/room/dataset-room1_512_16/mav0/cam0/data "$pathDatasetTUM_VI"/room/dataset-room1_512_16/mav0/cam1/data Stereo-Inertial/TUM_TimeStamps/dataset-room1_512.txt Stereo-Inertial/TUM_IMU/dataset-room1_512.txt "$pathDatasetTUM_VI"/magistrale/dataset-magistrale1_512_16/mav0/cam0/data "$pathDatasetTUM_VI"/magistrale/dataset-magistrale1_512_16/mav0/cam1/data Stereo-Inertial/TUM_TimeStamps/dataset-magistrale1_512.txt Stereo-Inertial/TUM_IMU/dataset-magistrale1_512.txt dataset-room1_magistrale1_512_stereoi