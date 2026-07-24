#!/bin/bash

machine=$(hostname)
sizes=("4000000")
modes=("fused" "split" "trace")
optimizations=("normal" "optimizations")
num_runs=5

if [[ "${machine}" == *"frontera"* ]]; then
    execution_spaces=("OpenMP")
    KP_READER="/work2/07159/nalawar/frontera/projects/fusion/kokkos-tools/kp_reader"
    output_file="overhead/frontera.csv"
    machine="frontera"
elif [[ "${machine}" == *"lassen"* ]]; then
    execution_spaces=("Cuda")
    KP_READER="/usr/workspace/alawar1/projects/pyk_lassen/kokkos-tools/kp_reader"
    output_file="overhead/lassen.csv"
    machine="lassen"
elif [[ "${machine}" == *"tioga"* ]]; then
    execution_spaces=("HIP")
    KP_READER="/usr/workspace/alawar1/projects/pyk_tioga/kokkos-tools/profiling/simple-kernel-timer/kp_reader"
    output_file="overhead/tioga.csv"
    machine="tioga"
elif [[ "${machine}" == *"ls6"* ]]; then
    execution_spaces=("OpenMP" "Cuda")
    KP_READER="/work/07159/nalawar/ls6/projects/fusion/kokkos-tools/kp_reader"
    output_file="overhead/lonestar.csv"
    machine="lonestar"
elif [[ "${machine}" == *"nader"* ]]; then
    execution_spaces=("OpenMP")
    KP_READER="/home/nader/projects/kokkos-tools/kp_reader"
    output_file="overhead/nader.csv"
    machine="nader"
fi

mkdir overhead

# timeout in seconds
readonly timeout=500

function pk.timeout() {
        local command=$1; shift

        command="/bin/sh -c \"$command\""

        expect -c "set echo \"-noecho\"; set timeout $timeout; spawn -noecho $command; expect timeout { exit 1 } eof { exit 0 }"

        if [ $? = 1 ] ; then
                # >&2 echo "Timeout after ${time} seconds"
                return 1
        fi
        return 0
}


for opt in "${optimizations[@]}"; do
    if [[ "${opt}" == "normal" ]]; then
        unset PK_RESTRICT
        unset PK_LOOP_FUSE
    elif [ "${opt}" == "optimizations" ]; then
        export PK_LOOP_FUSE=1
        export PK_RESTRICT=1
    fi

    for execution_space in "${execution_spaces[@]}"; do
        export PK_EXA_SPACE="${execution_space}"
        output_file="overhead/${machine}_${opt}_${execution_space}.csv"
        echo "benchmark,size,space,time,mode" > "${output_file}"

        for size in "${sizes[@]}"; do
            if [ "${execution_space}" == "OpenMP" ]; then
                size="1000000"
            fi

            for mode in "${modes[@]}"; do
                for i in $(seq $num_runs); do
                    if [[ "${mode}" == "fused" ]]; then
                        export PK_FUSION=naive
                        export PK_FUSE_ARGS=1
                    elif [[ "${mode}" == "trace" ]]; then
                        export PK_FUSION=trace
                        unset PK_FUSE_ARGS
                    else
                        unset PK_FUSION
                        unset PK_FUSE_ARGS
                    fi
                    command="python main.py -nw -s 400 -g 1200 -e 3.5 -N $size -space $execution_space"

                    echo "Running command ${command} with size ${size} ${opt} ${mode} ${execution_space}"

                    output=$( { time ${command}; } 2>&1 )
                    # format 0m1.525s
                    runtime=$(echo "$output" | grep real | awk '{print $2}' | cut -d'm' -f2 | cut -d's' -f1)

                    echo "boltzmann,${size},${execution_space},${runtime},${mode}" >> "${output_file}"
                done
            done
        done
    done
done