#!/bin/bash

machine=$(hostname)
sizes=("4000000")
modes=("fused" "split" "joined" "trace")
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
    execution_spaces=("Cuda" "OpenMP")
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
        output_file="results/${machine}_${opt}_${execution_space}.csv"
        overhead_file="overhead/${machine}_${opt}_${execution_space}.csv"
        overhead_timers_file="overhead/${machine}_${opt}_${execution_space}_timers.csv"
        echo "benchmark,size,space,time,mode" > "${output_file}"
        echo "benchmark,size,space,kernel_time,total_time,mode" > "${overhead_file}"
        echo "benchmark,size,space,runtime_tracing,runtime_fusion,compiler_fusion,mode" > "${overhead_timers_file}"

        for size in "${sizes[@]}"; do
            if [ "${execution_space}" == "OpenMP" ]; then
                size="1000000"
            fi
            for mode in "${modes[@]}"; do
                for i in $(seq $num_runs); do
                    echo "Running space ${execution_space} mode ${mode} run ${i}"
                    if [ "${mode}" == "split" ] || [ "${mode}" == "trace" ]; then
                        if [[ "${mode}" == "trace" ]]; then
                            export PK_FUSION=trace
                        fi

                        std_output=$(python main.py -nw -s 400 -g 1200 -e 3.5 -N $size -space $execution_space)

                        runtime_tracing_timer=$(echo "${std_output}" | grep "runtime tracing" | cut -d' ' -f3)
                        runtime_fusion_timer=$(echo "${std_output}" | grep "runtime fusion" | cut -d' ' -f3)
                        compiler_fusion_timer=$(echo "${std_output}" | grep "compiler fusion" | cut -d' ' -f3)

                        output=$(echo "${std_output}" | tail -n 3 | head -n 1)
                        split=$output
                        dat_file=$(echo "${std_output}" | tail -n 1 | cut -d' ' -f6)
                        dat_output=$(eval "${KP_READER} ${dat_file}")
                        xsection=$(echo "${dat_output}" | grep "xsection_kernel" -A 1 | tail -n 1 | cut -d' ' -f5)
                        collision=$(echo "${dat_output}" | grep "collision_kernel" -A 1 | tail -n 1 | cut -d' ' -f5)
                        advection=$(echo "${dat_output}" | grep "advection_kernel" -A 1 | tail -n 1 | cut -d' ' -f5)
                        kernel_time=$(echo "$xsection + $collision + $advection" | bc)

                        if [[ "${mode}" == "trace" ]]; then
                            unset PK_FUSION
                        fi
                    elif [ "${mode}" == "joined" ]; then
                        std_output=$(python main.py -nw -s 400 -g 1200 -e 3.5 -N $size -j -space $execution_space)

                        runtime_tracing_timer=$(echo "${std_output}" | grep "runtime tracing" | cut -d' ' -f3)
                        runtime_fusion_timer=$(echo "${std_output}" | grep "runtime fusion" | cut -d' ' -f3)
                        compiler_fusion_timer=$(echo "${std_output}" | grep "compiler fusion" | cut -d' ' -f3)

                        output=$(echo "${std_output}" | tail -n 3 | head -n 1)
                        joined=$output
                        dat_file=$(echo "${std_output}" | tail -n 1 | cut -d' ' -f6)
                        dat_output=$(eval "${KP_READER} ${dat_file}")
                        kernel_time=$(echo "${dat_output}" | grep "electron_kernel" -A 1 | tail -n 1 | cut -d' ' -f5)

                    else
                        export PK_FUSION=naive
                        export PK_FUSE_ARGS=1

                        std_output=$(python main.py -nw -s 400 -g 1200 -e 3.5 -N $size -space $execution_space)

                        runtime_tracing_timer=$(echo "${std_output}" | grep "runtime tracing" | cut -d' ' -f3)
                        runtime_fusion_timer=$(echo "${std_output}" | grep "runtime fusion" | cut -d' ' -f3)
                        compiler_fusion_timer=$(echo "${std_output}" | grep "compiler fusion" | cut -d' ' -f3)

                        output=$(echo "${std_output}" | tail -n 3 | head -n 1)
                        dat_file=$(echo "${std_output}" | tail -n 1 | cut -d' ' -f6)
                        dat_output=$(eval "${KP_READER} ${dat_file}")

                        unset PK_FUSION
                        unset PK_FUSE_ARGS

                        kernel_time=$(echo "${dat_output}" | grep "xsection" -A 1 | tail -n 1 | cut -d' ' -f5)
                    fi

                    total_time=$(echo "${dat_output}" | grep "Total Execution Time (incl. Kokkos + non-Kokkos):" | cut -d':' -f2 | xargs | cut -d' ' -f1)
                    total_kernel_time=$(echo "${dat_output}" | grep "Total Time in Kokkos kernels:" | cut -d':' -f2 | xargs | cut -d' ' -f1)

                    echo "Electron kernel,${size},${execution_space},${kernel_time},${mode}" >> "${output_file}"

                    echo "boltzmann,${size},${execution_space},${total_kernel_time},${total_time},${mode}" >> "${overhead_file}"
                    echo "boltzmann,${size},${execution_space},${runtime_tracing_timer},${runtime_fusion_timer},${compiler_fusion_timer},${mode}" >> "${overhead_timers_file}"
                done
            done
        done
    done
done

rm *.dat