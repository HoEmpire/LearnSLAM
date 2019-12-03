cmake_minimum_required(VERSION 2.8)
project(test)

set(CMAKE_BUILD_TYPE "Release")
add_definitions(-std=c++11 -Wall -O2)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(OpenCV 3 REQUIRED)
find_package(Sophus REQUIRED)

include_directories(
        ${OpenCV_INCLUDE_DIRS}
        "/usr/include/eigen3/"
)

add_executable(feature_matching ${PROJECT_SOURCE_DIR}/test/feature_matching.cpp)
target_link_libraries(feature_matching ${OpenCV_LIBS})

# add_executable( pose_estimation_2d2d pose_estimation_2d2d.cpp extra.cpp ) # use this if in OpenCV2 
#add_executable(pose_estimation_2d2d pose_estimation_2d2d.cpp)
#target_link_libraries(pose_estimation_2d2d ${OpenCV_LIBS})

# # add_executable( triangulation triangulation.cpp extra.cpp) # use this if in opencv2
#add_executable(triangulation triangulation.cpp)
#target_link_libraries(triangulation ${OpenCV_LIBS})

#add_executable(pose_estimation_3d2d pose_estimation_3d2d.cpp)
#target_link_libraries(pose_estimation_3d2d
#        g2o_core g2o_stuff
#        ${OpenCV_LIBS})

#add_executable(pose_estimation_3d3d pose_estimation_3d3d.cpp)
#target_link_libraries(pose_estimation_3d3d
#        g2o_core g2o_stuff
#        ${OpenCV_LIBS})
