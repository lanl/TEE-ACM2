#!/bin/bash
if [ $# -ne 12 ]
  then
    echo "usage : sh $0 <DIM_X> <DIM_Y> <BLK_M> <BLK_N> <BLK_K> <DIM_XA> <DIM_YA> <DIM_XB> <DIM_YB> <THR_M> <THR_N> <out_file>"
    exit 1
fi

DIM_X=$1
DIM_Y=$2
BLK_M=$3
BLK_N=$4
BLK_K=$5
DIM_XA=$6
DIM_YA=$7
DIM_XB=$8
DIM_YB=$9
THR_M=${10}
THR_N=${11}
output_file=${12}

out_n="mat_test_"$DIM_X"_"$DIM_Y"_"$BLK_M"_"$BLK_N"_"$BLK_K"_"$DIM_XA"_"$DIM_YA"_"$DIM_XB"_"$DIM_YB"_"$THR_M"_"$THR_N
echo $out_n

echo nvcc ../main.cu -o $output_file/$out_n -I/projects/darwin-nv/centos8/x86_64/packages/cuda/11.4.2/include \
 -L/projects/darwin-nv/centos8/x86_64/packages/cuda/11.4.2/targets/x86_64-linux/lib/stubs -lcuda -lcudart -lnvidia-ml -lcublas \
-DDIM_X=$DIM_X -DDIM_Y=$DIM_Y -DBLK_M=$BLK_M -DBLK_N=$BLK_N -DBLK_K=$BLK_K -DDIM_XA=$DIM_XA -DDIM_YA=$DIM_YA -DDIM_XB=$DIM_XB -DDIM_YB=$DIM_YB -DTHR_M=$THR_M -DTHR_N=$THR_N 

nvcc ../main.cu -o $output_file/$out_n -I/projects/darwin-nv/centos8/x86_64/packages/cuda/11.4.2/include \
 -L/projects/darwin-nv/centos8/x86_64/packages/cuda/11.4.2/targets/x86_64-linux/lib/stubs -lcuda -lcudart -lnvidia-ml -lcublas \
-DDIM_X=$DIM_X -DDIM_Y=$DIM_Y -DBLK_M=$BLK_M -DBLK_N=$BLK_N -DBLK_K=$BLK_K -DDIM_XA=$DIM_XA -DDIM_YA=$DIM_YA -DDIM_XB=$DIM_XB -DDIM_YB=$DIM_YB -DTHR_M=$THR_M -DTHR_N=$THR_N 