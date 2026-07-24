#!/bin/bash

machine=$(hostname)
sizes=("1000000" "2000000" "4000000" "8000000" "16000000")
modes=("fused" "split")
optimizations=("normal" "optimizations")
num_runs=5

if [[ "${machine}" == *"frontera"* ]]; then
    export KOKKOS_TOOLS_LIBS=/work2/07159/nalawar/frontera/projects/fusion/kokkos-tools/kp_kernel_timer.so
    execution_spaces=("OpenMP")
    KP_READER="/work2/07159/nalawar/frontera/projects/fusion/kokkos-tools/kp_reader"
    output_file="results/frontera.csv"
    machine="frontera"
elif [[ "${machine}" == *"lassen"* ]]; then
    export KOKKOS_TOOLS_LIBS=/usr/workspace/alawar1/projects/pyk_lassen/kokkos-tools/kp_kernel_timer.so
    execution_spaces=("Cuda")
    KP_READER="/usr/workspace/alawar1/projects/pyk_lassen/kokkos-tools/kp_reader"
    output_file="results/lassen.csv"
    machine="lassen"
elif [[ "${machine}" == *"tioga"* ]]; then
    export KOKKOS_TOOLS_LIBS=/usr/workspace/alawar1/projects/pyk_tioga/kokkos-tools/profiling/simple-kernel-timer/kp_kernel_timer.so
    execution_spaces=("HIP")
    KP_READER="/usr/workspace/alawar1/projects/pyk_tioga/kokkos-tools/profiling/simple-kernel-timer/kp_reader"
    output_file="results/tioga.csv"
    machine="tioga"
elif [[ "${machine}" == *"ls6"* ]]; then
    export KOKKOS_TOOLS_LIBS=/work/07159/nalawar/ls6/projects/fusion/kokkos-tools/kp_kernel_timer.so
    # execution_spaces=("Cuda" "OpenMP")
    execution_spaces=("Cuda")
    KP_READER="/work/07159/nalawar/ls6/projects/fusion/kokkos-tools/kp_reader"
    output_file="results/lonestar.csv"
    machine="lonestar"
fi

for opt in "${optimizations[@]}"; do
    if [[ "${opt}" == "normal" ]]; then
        unset PK_RESTRICT
        unset PK_LOOP_FUSE
    elif [ "${opt}" == "optimizations" ]; then
        export PK_LOOP_FUSE=1
        export PK_RESTRICT=1
    fi

    for execution_space in "${execution_spaces[@]}"; do
        for size in "${sizes[@]}"; do
            output_file="results/${machine}_${opt}_${execution_space}_${size}.csv"
            echo "benchmark,size,space,time,mode" > "${output_file}"

            for mode in "${modes[@]}"; do
                if [ "${mode}" == "split" ] && [ "${opt}" == "optimizations" ]; then
                    continue
                elif [ "${mode}" == "fused" ] && [ "${opt}" == "normal" ]; then
                    continue
                fi

                for i in $(seq $num_runs); do
                    echo "Running space ${execution_space} size ${size} mode ${mode} run ${i} optimizations ${opt}"
                    if [ "${mode}" == "split" ]; then
                        std_output=$(python main.py -nw -s 400 -g 1200 -e 3.5 -N $size -space $execution_space)
                        output=$(echo "${std_output}" | tail -n 3 | head -n 1)
                        split=$output
                        dat_file=$(echo "${std_output}" | tail -n 1 | cut -d' ' -f6)
                        dat_output=$(eval "${KP_READER} ${dat_file}")
                        xsection=$(echo "${dat_output}" | grep "xsection_kernel" -A 1 | tail -n 1 | cut -d' ' -f5)
                        collision=$(echo "${dat_output}" | grep "collision_kernel" -A 1 | tail -n 1 | cut -d' ' -f5)
                        advection=$(echo "${dat_output}" | grep "advection_kernel" -A 1 | tail -n 1 | cut -d' ' -f5)
                        kernel_time=$(echo "$xsection + $collision + $advection" | bc)

                    elif [ "${mode}" == "joined" ]; then
                        std_output=$(python main.py -nw -s 400 -g 1200 -e 3.5 -N $size -j -space $execution_space)
                        output=$(echo "${std_output}" | tail -n 3 | head -n 1)
                        joined=$output
                        dat_file=$(echo "${std_output}" | tail -n 1 | cut -d' ' -f6)
                        dat_output=$(eval "${KP_READER} ${dat_file}")
                        kernel_time=$(echo "${dat_output}" | grep "electron_kernel" -A 1 | tail -n 1 | cut -d' ' -f5)

                    else
                        export PK_FUSION=naive
                        export PK_FUSE_ARGS=1

                        std_output=$(python main.py -nw -s 400 -g 1200 -e 3.5 -N $size -space $execution_space)
                        output=$(echo "${std_output}" | tail -n 3 | head -n 1)
                        dat_file=$(echo "${std_output}" | tail -n 1 | cut -d' ' -f6)
                        dat_output=$(eval "${KP_READER} ${dat_file}")

                        unset PK_FUSION
                        unset PK_FUSE_ARGS

                        kernel_time=$(echo "${dat_output}" | grep "xsection" -A 1 | tail -n 1 | cut -d' ' -f5)
                    fi

                    echo "Electron kernel,${size},${execution_space},${kernel_time},${mode}" >> "${output_file}"
                done
            done
        done
    done
done

rm *.dat
