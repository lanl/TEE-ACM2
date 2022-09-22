#!/bin/bash
if [ $# -ne 2 ]
  then
    echo "usage : sh $0 <start_out_file> <z_dim>"
    exit 1
fi


# -P0 512 -P1 512 -P2 512 -x 128 -z 16 -y 128 -yBlockDim 1 -xBlockDim 256 -valid_res 1 -iter 1
DIM_X=16
DIM_Y=16

#TAB_BLK_MN=(16,16 16,32 192,192 208,208 224,224 240,240 )

#TAB_BLK_N=(16)

DIM_XA=16
DIM_YA=16
DIM_XB=16
DIM_YB=16
#THR_M=${10}
#THR_N=${11}

# FIXED z

BLK_K=$2
start_out=$1/z_$BLK_K/th_"$DIM_Y"_"$DIM_X"

THR_M=$((BLK_M / DIM_X))
THR_N=$((BLK_N / DIM_Y))

NB_REG_PER_BLOCK=$((THR_M*THR_N*DIM_X*DIM_Y))
S_MEM_PER_BLOCK=$((BLK_M*BLK_K + BLK_N*BLK_K))


part_nb=1
compiled_exe=0
mkdir -p $start_out/part_$part_nb

BLK_M=32
BLK_N=32

while [[ NB_REG_PER_BLOCK -le 65536 && S_MEM_PER_BLOCK -le 12888 ]]
do

  while [[ NB_REG_PER_BLOCK -le 65536 && S_MEM_PER_BLOCK -le 12888 ]]
  do
    if [[ $compiled_exe -ge 60 ]]
    then
      compiled_exe=0
      part_nb=$(( $part_nb + 1 ))
      mkdir -p $start_out/part_$part_nb
    fi

    echo sh compile_mat_test.sh $DIM_X $DIM_Y $BLK_M $BLK_N $BLK_K $DIM_XA $DIM_YA $DIM_XB $DIM_YB $THR_M $THR_N "$start_out/part_$part_nb"
    sh compile_mat_test.sh $DIM_X $DIM_Y $BLK_M $BLK_N $BLK_K $DIM_XA $DIM_YA $DIM_XB $DIM_YB $THR_M $THR_N "$start_out/part_$part_nb"

    retVal=$?
    if [ $retVal -eq 0 ]; then
      compiled_exe=$(( $compiled_exe + 1 ))
    fi
    
    BLK_N=$(( $BLK_N + 16 ))

    THR_M=$((BLK_M / DIM_X))
    THR_N=$((BLK_N / DIM_Y))

    NB_REG_PER_BLOCK=$((THR_M*THR_N*DIM_X*DIM_Y))
    S_MEM_PER_BLOCK=$((BLK_M*BLK_K + BLK_N*BLK_K))
  done
  
  BLK_N=32
  BLK_M=$(( $BLK_M + 16 ))

  THR_M=$((BLK_M / DIM_X))
  THR_N=$((BLK_N / DIM_Y))

  NB_REG_PER_BLOCK=$((THR_M*THR_N*DIM_X*DIM_Y))
  S_MEM_PER_BLOCK=$((BLK_M*BLK_K + BLK_N*BLK_K))
done

