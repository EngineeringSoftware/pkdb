#!/bin/bash

# There have been some issues with running ncu on lassen recently. The
# default ncu does not seem to be compatible with cuda/12.0 anymore,
# so we have to load a newer one. We can load a newer one through the
# nvhpc package, although that also loads an nvcc so you have to be
# careful. You also have to run the command `dcgmi profile --pause` to
# allow ncu to work properly.

machine=$(hostname)
size="4000000"
modes=("fused" "split" "joined")
optimizations=("normal" "optimizations")
metrics="smsp__sass_thread_inst_executed_op_fp64_pred_on.sum,smsp__sass_thread_inst_executed_op_integer_pred_on.sum,dram__bytes_read.sum,dram__bytes_write.sum,smsp__inst_executed_op_global_ld.sum,smsp__inst_executed_op_global_st.sum,dram__bytes_read.sum.per_second,dram__bytes_write.sum.per_second"

declare -A mode_kernels

split_kernels=("xsection_kernel_tag>" "collision_kernel_tag>" "advection_kernel_tag>")
fused_kernels=("xsection_kernel_collision_kernel_advection_kernel_tag>")
joined_kernels=("electron_kernel_1D_tag>")

mode_kernels["split"]="${split_kernels[@]}"
mode_kernels["fused"]="${fused_kernels[@]}"
mode_kernels["joined"]="${joined_kernels[@]}"

if [[ "${machine}" == *"frontera"* ]]; then
    execution_spaces=("OpenMP")
    machine="frontera"
elif [[ "${machine}" == *"lassen"* ]]; then
    execution_spaces=("Cuda")
    machine="lassen"
elif [[ "${machine}" == *"tioga"* ]]; then
    execution_spaces=("HIP")
    machine="tioga"
elif [[ "${machine}" == *"ls6"* ]]; then
    # execution_spaces=("Cuda" "OpenMP")
    execution_spaces=("Cuda")
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
        output_file="profile/${machine}_${opt}_${execution_space}.csv"
        echo "kernel,space,size,mode,bytes read,bytes written,loads,stores,fp64,int,read throughput,write throughput" > "${output_file}"

        for mode in "${modes[@]}"; do
            if [[ "${mode}" == "fused" ]]; then
                export PK_FUSION=naive
                export PK_FUSE_ARGS=1
            else
                unset PK_FUSION
                unset PK_FUSE_ARGS
            fi

            if [ "${execution_space}" == "Cuda" ]; then
                if [ "${mode}" == "joined" ]; then
                    command="ncu -k regex:cuda_parallel_launch* --csv --metrics '${metrics}' python main.py -nw -s 400 -g 1200 -e 3.5 -N $size -j -space $execution_space"
                else
                    command="ncu -k regex:cuda_parallel_launch* --csv --metrics '${metrics}' python main.py -nw -s 400 -g 1200 -e 3.5 -N $size -space $execution_space"
                fi
                output=$(eval "${command}" | sed -n '/^"ID"/,$p')

                kernels=(${mode_kernels["${mode}"]})

                for kernel in "${kernels[@]}"; do
                    #                                                                                get the last field        | remove the last character (" in this case") | remove all commas | get average
                    bytes_read=$(echo "${output}" | grep "dram__bytes_read.sum" | grep "${kernel}" | awk -F'","' '{print $15}' | sed 's/.$//' | sed 's/,//g' | awk '{ sum += $1; n++ } END { if (n > 0) print sum / n; }')
                    bytes_write=$(echo "${output}" | grep "dram__bytes_write.sum" | grep "${kernel}" | awk -F'","' '{print $15}' | sed 's/.$//' | sed 's/,//g' | awk '{ sum += $1; n++ } END { if (n > 0) print sum / n; }')
                    global_ld=$(echo "${output}" | grep "smsp__inst_executed_op_global_ld.sum" | grep "${kernel}" | awk -F'","' '{print $15}' | sed 's/.$//' | sed 's/,//g' | awk '{ sum += $1; n++ } END { if (n > 0) print sum / n; }')
                    global_st=$(echo "${output}" | grep "smsp__inst_executed_op_global_st.sum" | grep "${kernel}" | awk -F'","' '{print $15}' | sed 's/.$//' | sed 's/,//g' | awk '{ sum += $1; n++ } END { if (n > 0) print sum / n; }')
                    fp64_inst=$(echo "${output}" | grep "smsp__sass_thread_inst_executed_op_fp64_pred_on.sum" | grep "${kernel}" | awk -F'","' '{print $15}' | sed 's/.$//' | sed 's/,//g' | awk '{ sum += $1; n++ } END { if (n > 0) print sum / n; }')
                    int_inst=$(echo "${output}" | grep "smsp__sass_thread_inst_executed_op_integer_pred_on.sum" | grep "${kernel}" | awk -F'","' '{print $15}' | sed 's/.$//' | sed 's/,//g' | awk '{ sum += $1; n++ } END { if (n > 0) print sum / n; }')
                    read_throughput=$(echo "${output}" | grep "dram__bytes_read.sum.per_second" | grep "${kernel}" | awk -F'","' '{print $15}' | sed 's/.$//' | sed 's/,//g' | awk '{ sum += $1; n++ } END { if (n > 0) print sum / n; }')
                    write_throughput=$(echo "${output}" | grep "dram__bytes_write.sum.per_second" | grep "${kernel}" | awk -F'","' '{print $15}' | sed 's/.$//' | sed 's/,//g' | awk '{ sum += $1; n++ } END { if (n > 0) print sum / n; }')

                    kernel_name=$(echo "${kernel}" | sed 's/_tag>//')
                    echo "${kernel_name},${execution_space},${size},${mode},${bytes_read},${bytes_write},${global_ld},${global_st},${fp64_inst},${int_inst},${read_throughput},${write_throughput}" >> "${output_file}"
                done

            elif [ "${execution_space}" == "HIP" ]; then
                export PK_BOLTZ_EARLY_EXIT=1
                if [ "${mode}" == "joined" ]; then
                    # ../../../omniperf_install/1.0.10/bin/omniperf profile -n "${mode}_${opt}" --device 0 -d 149 --no-roof -- /usr/workspace/alawar1/projects/anaconda_tioga/envs/pyk_tioga/bin/python main.py -nw -s 400 -g 1200 -e 3.5 -N "${size}" -space "${execution_space}" -j
                    output_0=$(eval "../../../omniperf_install/1.0.10/bin/omniperf analyze -p workloads/${mode}_${opt}/mi200/ -n per_kernel -b 10.2 10.3 17.2")

                    bytes_read=$(echo "${output_0}" | grep "17\.2\.0 " | cut -d'│' -f4 | xargs) # xargs strips whitespace
                    bytes_write=$(echo "${output_0}" | grep "17\.2\.1 " | cut -d'│' -f4 | xargs)
                    global_ld=$(echo "${output_0}" | grep "10\.3\.5 " | cut -d'│' -f4 | xargs)
                    global_st=$(echo "${output_0}" | grep "10\.3\.6 " | cut -d'│' -f4 | xargs)
                    fp64_add_inst=$(echo "${output_0}" | grep "10\.2\.10 " | cut -d'│' -f4 | xargs)
                    fp64_mul_inst=$(echo "${output_0}" | grep "10\.2\.11 " | cut -d'│' -f4 | xargs)
                    fp64_fma_inst=$(echo "${output_0}" | grep "10\.2\.12 " | cut -d'│' -f4 | xargs)
                    fp64_trans_inst=$(echo "${output_0}" | grep "10\.2\.13 " | cut -d'│' -f4 | xargs)
                    fp64_inst=$(echo "$fp64_add_inst + $fp64_mul_inst + $fp64_fma_inst + $fp64_trans_inst" | bc)
                    int32_inst=$(echo "${output_0}" | grep "10\.2\.0 " | cut -d'│' -f4 | xargs)
                    int64_inst=$(echo "${output_0}" | grep "10\.2\.1 " | cut -d'│' -f4 | xargs)
                    int_inst=$(echo "$int32_inst + $int64_inst" | bc)

                    echo "electron_kernel_1D,${execution_space},${size},${mode},${bytes_read},${bytes_write},${global_ld},${global_st},${fp64_inst},${int_inst},-1,-1" >> "${output_file}"

                elif [ "${mode}" == "split" ]; then
                    # ../../../omniperf_install/1.0.10/bin/omniperf profile -n "${mode}_${opt}_0" --device 0 -d 149 --no-roof -- /usr/workspace/alawar1/projects/anaconda_tioga/envs/pyk_tioga/bin/python main.py -nw -s 400 -g 1200 -e 3.5 -N "${size}" -space "${execution_space}"
                    output_0=$(eval "../../../omniperf_install/1.0.10/bin/omniperf analyze -p workloads/${mode}_${opt}_0/mi200/ -n per_kernel -b 10.2 10.3 17.2")
                    bytes_read_0=$(echo "${output_0}" | grep "17\.2\.0 " | cut -d'│' -f4 | xargs) # xargs strips whitespace
                    bytes_write_0=$(echo "${output_0}" | grep "17\.2\.1 " | cut -d'│' -f4 | xargs)
                    global_ld_0=$(echo "${output_0}" | grep "10\.3\.5 " | cut -d'│' -f4 | xargs)
                    global_st_0=$(echo "${output_0}" | grep "10\.3\.6 " | cut -d'│' -f4 | xargs)

                    fp64_add_inst=$(echo "${output_0}" | grep "10\.2\.10 " | cut -d'│' -f4 | xargs)
                    fp64_mul_inst=$(echo "${output_0}" | grep "10\.2\.11 " | cut -d'│' -f4 | xargs)
                    fp64_fma_inst=$(echo "${output_0}" | grep "10\.2\.12 " | cut -d'│' -f4 | xargs)
                    fp64_trans_inst=$(echo "${output_0}" | grep "10\.2\.13 " | cut -d'│' -f4 | xargs)
                    fp64_inst_0=$(echo "$fp64_add_inst + $fp64_mul_inst + $fp64_fma_inst + $fp64_trans_inst" | bc)
                    int32_inst=$(echo "${output_0}" | grep "10\.2\.0 " | cut -d'│' -f4 | xargs)
                    int64_inst=$(echo "${output_0}" | grep "10\.2\.1 " | cut -d'│' -f4 | xargs)
                    int_inst_0=$(echo "$int32_inst + $int64_inst" | bc)

                    # ../../../omniperf_install/1.0.10/bin/omniperf profile -n "${mode}_${opt}_1" --device 0 -d 150 --no-roof -- /usr/workspace/alawar1/projects/anaconda_tioga/envs/pyk_tioga/bin/python main.py -nw -s 400 -g 1200 -e 3.5 -N "${size}" -space "${execution_space}"
                    output_1=$(eval "../../../omniperf_install/1.0.10/bin/omniperf analyze -p workloads/${mode}_${opt}_1/mi200/ -n per_kernel -b 10.2 10.3 17.2")
                    bytes_read_1=$(echo "${output_1}" | grep "17\.2\.0 " | cut -d'│' -f4 | xargs) # xargs strips whitespace
                    bytes_write_1=$(echo "${output_1}" | grep "17\.2\.1 " | cut -d'│' -f4 | xargs)
                    global_ld_1=$(echo "${output_1}" | grep "10\.3\.5 " | cut -d'│' -f4 | xargs)
                    global_st_1=$(echo "${output_1}" | grep "10\.3\.6 " | cut -d'│' -f4 | xargs)

                    fp64_add_inst=$(echo "${output_1}" | grep "10\.2\.10 " | cut -d'│' -f4 | xargs)
                    fp64_mul_inst=$(echo "${output_1}" | grep "10\.2\.11 " | cut -d'│' -f4 | xargs)
                    fp64_fma_inst=$(echo "${output_1}" | grep "10\.2\.12 " | cut -d'│' -f4 | xargs)
                    fp64_trans_inst=$(echo "${output_1}" | grep "10\.2\.13 " | cut -d'│' -f4 | xargs)
                    fp64_inst_1=$(echo "$fp64_add_inst + $fp64_mul_inst + $fp64_fma_inst + $fp64_trans_inst" | bc)
                    int32_inst=$(echo "${output_1}" | grep "10\.2\.0 " | cut -d'│' -f4 | xargs)
                    int64_inst=$(echo "${output_1}" | grep "10\.2\.1 " | cut -d'│' -f4 | xargs)
                    int_inst_1=$(echo "$int32_inst + $int64_inst" | bc)

                    # ../../../omniperf_install/1.0.10/bin/omniperf profile -n "${mode}_${opt}_2" --device 0 -d 151 --no-roof -- /usr/workspace/alawar1/projects/anaconda_tioga/envs/pyk_tioga/bin/python main.py -nw -s 400 -g 1200 -e 3.5 -N "${size}" -space "${execution_space}"
                    output_2=$(eval "../../../omniperf_install/1.0.10/bin/omniperf analyze -p workloads/${mode}_${opt}_2/mi200/ -n per_kernel -b 10.2 10.3 17.2")
                    bytes_read_2=$(echo "${output_2}" | grep "17\.2\.0 " | cut -d'│' -f4 | xargs) # xargs strips whitespace
                    bytes_write_2=$(echo "${output_2}" | grep "17\.2\.1 " | cut -d'│' -f4 | xargs)
                    global_ld_2=$(echo "${output_2}" | grep "10\.3\.5 " | cut -d'│' -f4 | xargs)
                    global_st_2=$(echo "${output_2}" | grep "10\.3\.6 " | cut -d'│' -f4 | xargs)

                    fp64_add_inst=$(echo "${output_2}" | grep "10\.2\.10 " | cut -d'│' -f4 | xargs)
                    fp64_mul_inst=$(echo "${output_2}" | grep "10\.2\.11 " | cut -d'│' -f4 | xargs)
                    fp64_fma_inst=$(echo "${output_2}" | grep "10\.2\.12 " | cut -d'│' -f4 | xargs)
                    fp64_trans_inst=$(echo "${output_2}" | grep "10\.2\.13 " | cut -d'│' -f4 | xargs)
                    fp64_inst_2=$(echo "$fp64_add_inst + $fp64_mul_inst + $fp64_fma_inst + $fp64_trans_inst" | bc)
                    int32_inst=$(echo "${output_2}" | grep "10\.2\.0 " | cut -d'│' -f4 | xargs)
                    int64_inst=$(echo "${output_2}" | grep "10\.2\.1 " | cut -d'│' -f4 | xargs)
                    int_inst_2=$(echo "$int32_inst + $int64_inst" | bc)

                    echo "xsection_kernel,${execution_space},${size},${mode},${bytes_read_0},${bytes_write_0},${global_ld_0},${global_st_0},${fp64_inst_0},${int_inst_0},-1,-1" >> "${output_file}"
                    echo "collision_kernel,${execution_space},${size},${mode},${bytes_read_1},${bytes_write_1},${global_ld_1},${global_st_1},${fp64_inst_1},${int_inst_1},-1,-1" >> "${output_file}"
                    echo "advection_kernel,${execution_space},${size},${mode},${bytes_read_2},${bytes_write_2},${global_ld_2},${global_st_2},${fp64_inst_2},${int_inst_2},-1,-1" >> "${output_file}"

                elif [ "${mode}" == "fused" ]; then
                    # ../../../omniperf_install/1.0.10/bin/omniperf profile -n "${mode}_${opt}" --device 0 -d 149 --no-roof -- /usr/workspace/alawar1/projects/anaconda_tioga/envs/pyk_tioga/bin/python main.py -nw -s 400 -g 1200 -e 3.5 -N "${size}" -space "${execution_space}"
                    output_0=$(eval "../../../omniperf_install/1.0.10/bin/omniperf analyze -p workloads/${mode}_${opt}/mi200/ -n per_kernel -b 10.2 10.3 17.2")
                    bytes_read=$(echo "${output_0}" | grep "17\.2\.0 " | cut -d'│' -f4 | xargs) # xargs strips whitespace
                    bytes_write=$(echo "${output_0}" | grep "17\.2\.1 " | cut -d'│' -f4 | xargs)
                    global_ld=$(echo "${output_0}" | grep "10\.3\.5 " | cut -d'│' -f4 | xargs)
                    global_st=$(echo "${output_0}" | grep "10\.3\.6 " | cut -d'│' -f4 | xargs)
                    fp64_add_inst=$(echo "${output_0}" | grep "10\.2\.10 " | cut -d'│' -f4 | xargs)
                    fp64_mul_inst=$(echo "${output_0}" | grep "10\.2\.11 " | cut -d'│' -f4 | xargs)
                    fp64_fma_inst=$(echo "${output_0}" | grep "10\.2\.12 " | cut -d'│' -f4 | xargs)
                    fp64_trans_inst=$(echo "${output_0}" | grep "10\.2\.13 " | cut -d'│' -f4 | xargs)
                    fp64_inst=$(echo "$fp64_add_inst + $fp64_mul_inst + $fp64_fma_inst + $fp64_trans_inst" | bc)
                    int32_inst=$(echo "${output_0}" | grep "10\.2\.0 " | cut -d'│' -f4 | xargs)
                    int64_inst=$(echo "${output_0}" | grep "10\.2\.1 " | cut -d'│' -f4 | xargs)
                    int_inst=$(echo "$int32_inst + $int64_inst" | bc)

                    echo "xsection_kernel_collision_kernel_advection_kernel,${execution_space},${size},${mode},${bytes_read},${bytes_write},${global_ld},${global_st},${fp64_inst},${int_inst},-1,-1" >> "${output_file}"
                fi

                unset PK_BOLTZ_EARLY_EXIT
            fi
        done
    done
done
