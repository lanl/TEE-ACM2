cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(mat_mult_energy_consumption LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

#
# 'main' executable
#
add_executable(main
  main.cu
  )

set_target_properties(main PROPERTIES CUDA_ARCHITECTURES "70")
target_link_libraries(main ${CUDA_LIBRARIES})
target_link_libraries(main -lcublas)
target_include_directories(main PRIVATE /projects/darwin-nv/centos8/x86_64/packages/cuda/11.4.2/include)

link_directories(/projects/darwin-nv/centos8/x86_64/packages/cuda/11.4.2/targets/x86_64-linux/lib/stubs)
target_link_libraries(main nvidia-ml)
