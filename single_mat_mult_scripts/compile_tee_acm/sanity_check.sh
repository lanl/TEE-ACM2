#!/bin/bash
if [ $# -ne 1 ]
  then
    echo "usage : sh $0 <path>"
    exit 1
fi

path=$1
for mat_exe in $path/mat*; do
  echo "################"
  echo "# $mat_exe"
  ./$mat_exe -P0 512 -P1 512 -P2 512 -x 128 -z 16 -y 128 -yBlockDim 1 -xBlockDim 256 -valid_res 1 -iter 1 -e_out_cublas tmp1.csv -e_out_tee_acm tmp2.csv
  retVal=$?
  if [ $retVal -ne 0 ]; then
    rm $mat_exe
    echo "$mat_exe removed"
  fi
done
#16_16_256_176_16_16_16_16_16_16_11