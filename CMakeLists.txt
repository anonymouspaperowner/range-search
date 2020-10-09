cmake_minimum_required(VERSION 2.8)
project(ss-interval)

#set(CMAKE_C_COMPILER /data/opt/gcc-9.1.0-bottle/bin/gcc)
#set(CMAKE_CXX_COMPILER /data/opt/gcc-9.1.0-bottle/bin/g++)

IF (CMAKE_BUILD_TYPE STREQUAL Debug)
    ADD_DEFINITIONS(-DDEBUG)
ENDIF()


find_package( PkgConfig )

if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()


include_directories(include)
SET( CMAKE_CXX_FLAGS  "-ftemplate-depth=1024 -Ofast -lrt -DNDEBUG -std=c++11 -DHAVE_CXX0X -openmp -march=native -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0" )


add_executable(main main.cpp)
add_executable(vq search_vq.cc)
add_executable(bf search_bf.cc)
add_executable(lsh search_lsh.cc)
add_executable(wlsh search_lsh_weighted.cc)
add_executable(kdt search_kdt.cc)
add_executable(rtree search_rtree.cc)
