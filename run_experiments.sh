#!/bin/bash


if [ $# -ne 2 ]; then
    echo "Usage: $0 <version> <num_iterations>"
    exit 1
fi

version=$1
num_itr=$2

# datasets=( "MH01" "MH02" "MH03" "MH04" "MH05" "V101" "V102" "V103" "V201" "V202" "V203" "room1" "room2" "room3" "room4" "room5" "room6" "corridor1" "corridor2" "corridor3")
datasets=("MH01" "MH02" "V101" "V102" "room1" "room2" "corridor1" "corridor2")

for i in $(seq 0 $(expr $num_itr - 1)); do
    for dataset in "${datasets[@]}"; do
        echo -e "[bash:] -> $dataset 0 0 0 0 pose_optimization=on $version.$i"
        ./run_script.sh 0 0 0 0 1 $dataset $version.$i

        echo -e "[bash:] -> $dataset 0 0 0 0 pose_optimization=off $version.$i"
        ./run_script.sh 0 0 0 0 0 $dataset $version.$i
    done
done


# for i in $(seq 0 $(expr $num_itr - 1)); do
#     for dataset in "${datasets[@]}"; do
#         echo -e "[bash:] -> $dataset 0 0 0 0 pose_optimization=on $version.$i"
#         ./run_script.sh 0 0 0 0 1 $dataset $version.$i

#         echo -e "[bash:] -> $dataset 1 1 1 1 pose_optimization=off $version.$i"
#         ./run_script.sh 1 1 1 1 0 $dataset $version.$i

        # echo -e "[bash:] -> $dataset 0 0 0 1 pose_optimization=on $version.$i"
        # ./run_script.sh 0 0 0 1 1 $dataset $version.$i

        # echo -e "[bash:] -> $dataset 0 0 1 0 pose_optimization=on $version.$i"
        # ./run_script.sh 0 0 1 0 1 $dataset $version.$i

        # echo -e "[bash:] -> $dataset 0 0 1 1 pose_optimization=on $version.$i"
        # ./run_script.sh 0 0 1 1 1 $dataset $version.$i

        # echo -e "[bash:] -> $dataset 0 1 0 0 pose_optimization=on $version.$i"
        # ./run_script.sh 0 1 0 0 1 $dataset $version.$i

        # echo -e "[bash:] -> $dataset 0 1 1 1 pose_optimization=on $version.$i"
        # ./run_script.sh 0 1 1 1 1 $dataset $version.$i

        # echo -e "[bash:] -> $dataset 1 0 0 0 pose_optimization=on $version.$i"
        # ./run_script.sh 1 0 0 0 1 $dataset $version.$i

        # echo -e "[bash:] -> $dataset 1 1 0 0 pose_optimization=on $version.$i"
        # ./run_script.sh 1 1 0 0 1 $dataset $version.$i
        
        # echo -e "[bash:] -> $dataset 1 0 1 1 pose_optimization=on $version.$i"
        # ./run_script.sh 1 0 1 1 1 $dataset $version.$i

        # echo -e "[bash:] -> $dataset 1 1 1 1 pose_optimization=on $version.$i"
        # ./run_script.sh 1 1 1 1 1 $dataset $version.$i



        # echo -e "[bash:] -> $dataset 0 0 0 0 pose_optimization=off $version.$i"
        # ./run_script.sh 0 0 0 0 0 $dataset $version.$i

        # echo -e "[bash:] -> $dataset 0 0 0 1 pose_optimization=off $version.$i"
        # ./run_script.sh 0 0 0 1 0 $dataset $version.$i

        # echo -e "[bash:] -> $dataset 0 0 1 0 pose_optimization=off $version.$i"
        # ./run_script.sh 0 0 1 0 0 $dataset $version.$i

        # echo -e "[bash:] -> $dataset 0 0 1 1 pose_optimization=off $version.$i"
        # ./run_script.sh 0 0 1 1 0 $dataset $version.$i

        # echo -e "[bash:] -> $dataset 0 1 0 0 pose_optimization=off $version.$i"
        # ./run_script.sh 0 1 0 0 0 $dataset $version.$i

        # echo -e "[bash:] -> $dataset 0 1 1 1 pose_optimization=off $version.$i"
        # ./run_script.sh 0 1 1 1 0 $dataset $version.$i

        # echo -e "[bash:] -> $dataset 1 0 0 0 pose_optimization=off $version.$i"
        # ./run_script.sh 1 0 0 0 0 $dataset $version.$i

        # echo -e "[bash:] -> $dataset 1 1 0 0 pose_optimization=off $version.$i"
        # ./run_script.sh 1 1 0 0 0 $dataset $version.$i
        
        # echo -e "[bash:] -> $dataset 1 0 1 1 pose_optimization=off $version.$i"
        # ./run_script.sh 1 0 1 1 0 $dataset $version.$i

        # echo -e "[bash:] -> $dataset 1 1 1 1 pose_optimization=off $version.$i"
        # ./run_script.sh 1 1 1 1 0 $dataset $version.$i
    # done
# done

# for dataset in "${datasets[@]}"; do
#     python3 calculate_average_results.py Results/poseOptimization_on/0000/$dataset Results_average/$version/$dataset/poseOptimization_on/0000 $version

#     python3 calculate_average_results.py Results/poseOptimization_off/1111/$dataset Results_average/$version/$dataset/poseOptimization_off/1111 $version

    # python3 calculate_average_results.py Results/poseOptimization_on/0001/$dataset Results_average/$version/$dataset/poseOptimization_on/0001 $version

    # python3 calculate_average_results.py Results/poseOptimization_on/0010/$dataset Results_average/$version/$dataset/poseOptimization_on/0010 $version

    # python3 calculate_average_results.py Results/poseOptimization_on/0011/$dataset Results_average/$version/$dataset/poseOptimization_on/0011 $version

    # python3 calculate_average_results.py Results/poseOptimization_on/0100/$dataset Results_average/$version/$dataset/poseOptimization_on/0100 $version

    # python3 calculate_average_results.py Results/poseOptimization_on/0111/$dataset Results_average/$version/$dataset/poseOptimization_on/0111 $version

    # python3 calculate_average_results.py Results/poseOptimization_on/1000/$dataset Results_average/$version/$dataset/poseOptimization_on/1000 $version

    # python3 calculate_average_results.py Results/poseOptimization_on/1100/$dataset Results_average/$version/$dataset/poseOptimization_on/1100 $version
    
    # python3 calculate_average_results.py Results/poseOptimization_on/1011/$dataset Results_average/$version/$dataset/poseOptimization_on/0010 $version

    # python3 calculate_average_results.py Results/poseOptimization_on/1111/$dataset Results_average/$version/$dataset/poseOptimization_on/1111 $version

    # python3 plot_comparison.py Results_average/$version/$dataset/poseOptimization_on Results_average_plots/$version/$dataset/poseOptimization_on



    # python3 calculate_average_results.py Results/poseOptimization_off/0000/$dataset Results_average/$version/$dataset/poseOptimization_off/0000 $version

    # python3 calculate_average_results.py Results/poseOptimization_off/0001/$dataset Results_average/$version/$dataset/poseOptimization_off/0001 $version

    # python3 calculate_average_results.py Results/poseOptimization_off/0010/$dataset Results_average/$version/$dataset/poseOptimization_off/0010 $version

    # python3 calculate_average_results.py Results/poseOptimization_off/0011/$dataset Results_average/$version/$dataset/poseOptimization_off/0011 $version

    # python3 calculate_average_results.py Results/poseOptimization_off/0100/$dataset Results_average/$version/$dataset/poseOptimization_off/0100 $version

    # python3 calculate_average_results.py Results/poseOptimization_off/0111/$dataset Results_average/$version/$dataset/poseOptimization_off/0111 $version

    # python3 calculate_average_results.py Results/poseOptimization_off/1000/$dataset Results_average/$version/$dataset/poseOptimization_off/1000 $version

    # python3 calculate_average_results.py Results/poseOptimization_off/1100/$dataset Results_average/$version/$dataset/poseOptimization_off/1100 $version
    
    # python3 calculate_average_results.py Results/poseOptimization_off/1011/$dataset Results_average/$version/$dataset/poseOptimization_off/0010 $version

    # python3 calculate_average_results.py Results/poseOptimization_off/1111/$dataset Results_average/$version/$dataset/poseOptimization_off/1111 $version

    # python3 plot_comparison.py Results_average/$version/$dataset/poseOptimization_off Results_average_plots/$version/$dataset/poseOptimization_off
# done