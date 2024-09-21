#!/bin/bash
# pathDatasetEuroc='../EuRoC-Dataset' #Example, it is necesary to change it by the dataset path
pathDatasetEuroc=$HOME/SLAM/Datasets/EuRoc

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

# Single Session Example (Pure visual)
# echo "Launching MH01 with Stereo sensor"
# ./Examples/Stereo/stereo_euroc ./Vocabulary/ORBvoc.txt ./Examples/Stereo/EuRoC.yaml "$pathDatasetEuroc"/MH01 ./Examples/Stereo/EuRoC_TimeStamps/MH01.txt dataset-MH01_stereo
# echo "------------------------------------"
# echo "Evaluation of MH01 trajectory with Stereo sensor"
# python ../evaluation/evaluate_ate_scale.py ../evaluation/Ground_truth/EuRoC_left_cam/MH01_GT.txt f_dataset-MH01_stereo.txt --plot MH01_stereo.pdf


# MultiSession Example (Pure visual)
# echo "Launching Machine Hall with Stereo sensor"
# ./Examples/Stereo/stereo_euroc ./Vocabulary/ORBvoc.txt ./Examples/Stereo/EuRoC.yaml "$pathDatasetEuroc"/MH01 ./Examples/Stereo/EuRoC_TimeStamps/MH01.txt "$pathDatasetEuroc"/MH02 ./Examples/Stereo/EuRoC_TimeStamps/MH02.txt "$pathDatasetEuroc"/MH03 ./Examples/Stereo/EuRoC_TimeStamps/MH03.txt "$pathDatasetEuroc"/MH04 ./Examples/Stereo/EuRoC_TimeStamps/MH04.txt "$pathDatasetEuroc"/MH05 ./Examples/Stereo/EuRoC_TimeStamps/MH05.txt dataset-MH01_to_MH05_stereo
# echo "------------------------------------"
# echo "Evaluation of MAchine Hall trajectory with Stereo sensor"
# python evaluation/evaluate_ate_scale.py evaluation/Ground_truth/EuRoC_left_cam/MH_GT.txt f_dataset-MH01_to_MH05_stereo.txt --plot MH01_to_MH05_stereo.pdf


# Single Session Example (Visual-Inertial)
# echo "Launching V102 with Monocular-Inertial sensor"
# ./Examples/Monocular-Inertial/mono_inertial_euroc ./Vocabulary/ORBvoc.txt ./Examples/Monocular-Inertial/EuRoC.yaml "$pathDatasetEuroc"/V102 ./Examples/Monocular-Inertial/EuRoC_TimeStamps/V102.txt dataset-V102_monoi
# echo "------------------------------------"
# echo "Evaluation of V102 trajectory with Monocular-Inertial sensor"
# python evaluation/evaluate_ate_scale.py "$pathDatasetEuroc"/V102/mav0/state_groundtruth_estimate0/data.csv f_dataset-V102_monoi.txt --plot V102_monoi.pdf

#Single Session Example (Stereo-Inertial)

echo "Launching $dataset_name with Stereo-Inertial sensor"
file_name="dataset-${dataset_name}_stereoi"
# EXECUTABLE=./Stereo-Inertial/stereo_inertial_euroc
# ARGS="../Vocabulary/ORBvoc.txt ./Stereo-Inertial/EuRoC.yaml "${pathDatasetEuroc}"/"${dataset_name}" ./Stereo-Inertial/EuRoC_TimeStamps/${dataset_name}.txt "${file_name}" "${statsDir}" ${orbExtractionRunstatus} ${stereoMatchRunstatus} ${searchLocalPointsRunstatus} ${poseEstimationRunstatus}"
# gdb -ex "set args $ARGS" -ex "run" $EXECUTABLE
./Stereo-Inertial/stereo_inertial_euroc ../Vocabulary/ORBvoc.txt ./Stereo-Inertial/EuRoC.yaml "${pathDatasetEuroc}"/"${dataset_name}" ./Stereo-Inertial/EuRoC_TimeStamps/${dataset_name}.txt "${file_name}" "${statsDir}" ${orbExtractionRunstatus} ${stereoMatchRunstatus} ${searchLocalPointsRunstatus} ${poseEstimationRunstatus} ${poseOptimizationRunstatus}

# EXECUTABLE=./Stereo/stereo_euroc 
# ARGS="../Vocabulary/ORBvoc.txt ./Stereo-Inertial/EuRoC.yaml "${pathDatasetEuroc}"/"${dataset_name}" ./Stereo-Inertial/EuRoC_TimeStamps/${dataset_name}.txt "${file_name}" "${statsDir}" ${orbExtractionRunstatus} ${stereoMatchRunstatus} ${searchLocalPointsRunstatus} ${poseEstimationRunstatus}"
# gdb -ex "set args $ARGS" -ex "run" $EXECUTABLE
# ./Stereo/stereo_euroc ../Vocabulary/ORBvoc.txt ./Stereo-Inertial/EuRoC.yaml "${pathDatasetEuroc}"/"${dataset_name}" ./Stereo-Inertial/EuRoC_TimeStamps/${dataset_name}.txt "${file_name}" "${statsDir}" ${orbExtractionRunstatus} ${stereoMatchRunstatus} ${searchLocalPointsRunstatus} ${poseEstimationRunstatus}
echo "------------------------------------"

echo "Evaluation of ${dataset_name} trajectory with Stereo-Inertial sensor"
python3 -W ignore ../evaluation/evaluate3.py ${pathDatasetEuroc}/${dataset_name}/mav0/state_groundtruth_estimate0/data.csv f_${file_name}.txt --plot ${dataset_name}_stereoi.pdf --verbose
# echo "Plotting data"
# python3 ../plot.py "${statsDir}"
# python3 ../plot_piechart.py "${statsDir}"

files=("f_dataset-${dataset_name}_stereoi.csv"
"f_dataset-${dataset_name}_stereoi.txt"
"f_dataset-${dataset_name}_stereoi.png"
"kf_dataset-${dataset_name}_stereoi.txt"
)
destination_directory="${statsDir}/trajectory"
mkdir -p $destination_directory
mv "${files[@]}" "$destination_directory"


# visual only
# ./Stereo/stereo_euroc ../Vocabulary/ORBvoc.txt ./Stereo/EuRoC.yaml "$pathDatasetEuroc"/MH01 ./Stereo/EuRoC_TimeStamps/MH01.txt dataset-MH01_stereo


# MultiSession Monocular Examples
# echo "Launching Vicon Room 2 with Monocular-Inertial sensor"
# ./Examples/Monocular-Inertial/mono_inertial_euroc ./Vocabulary/ORBvoc.txt ./Examples/Monocular-Inertial/EuRoC.yaml "$pathDatasetEuroc"/V201 ./Examples/Monocular-Inertial/EuRoC_TimeStamps/V201.txt "$pathDatasetEuroc"/V202 ./Examples/Monocular-Inertial/EuRoC_TimeStamps/V202.txt "$pathDatasetEuroc"/V203 ./Examples/Monocular-Inertial/EuRoC_TimeStamps/V203.txt dataset-V201_to_V203_monoi
# echo "------------------------------------"
# echo "Evaluation of Vicon Room 2 trajectory with Stereo sensor"
# python evaluation/evaluate_ate_scale.py evaluation/Ground_truth/EuRoC_imu/V2_GT.txt f_dataset-V201_to_V203_monoi.txt --plot V201_to_V203_monoi.pdf