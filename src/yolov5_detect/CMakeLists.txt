cmake_minimum_required(VERSION 3.0.2)
project(yolov5_detect)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  message_generation
  roscpp
  rospy
  std_msgs
  std_srvs
  OpenCV
  cv_bridge
  image_transport
)

add_message_files(
  FILES
  detect.msg
  # Message2.msg
)

#generate_messages必须在catkin_package前面
generate_messages(
 DEPENDENCIES
 geometry_msgs
)

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
  CATKIN_DEPENDS message_runtime
  INCLUDE_DIRS include
#  LIBRARIES yolov5_detect
#  CATKIN_DEPENDS actionlib geometry_msgs mavros mavros_msgs message_generation nav_msgs pluginlib roscpp rospy seneor_msgs std_msgs std_srvs tf2_eigen tf2_ros trajectory_msgs
#  DEPENDS system_lib
)


###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include 
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_definitions(-std=c++11)
add_definitions(-DAPI_EXPORTS)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)

if(WIN32)
enable_language(CUDA)
endif(WIN32)

# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

# tensorrt
include_directories(/usr/include/x86_64-linux-gnu/)
link_directories(/usr/lib/x86_64-linux-gnu/)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -g -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED -pthread")
cuda_add_library(myplugins SHARED src/yololayer.cu)
target_link_libraries(myplugins nvinfer cudart)

# find_package(OpenCV)
# include_directories(${OpenCV_INCLUDE_DIRS})

add_library(sort src/sort.cpp src/Hungarian.cpp src/KalmanTracker.cpp)

cuda_add_executable(yolov5_detect src/calibrator.cpp src/yolov5_detect_1920.cpp src/preprocess.cu src/video.cpp)

target_link_libraries(yolov5_detect nvinfer)
target_link_libraries(yolov5_detect cudart)
target_link_libraries(yolov5_detect myplugins)
target_link_libraries(yolov5_detect ${OpenCV_LIBS})
target_link_libraries(yolov5_detect ${catkin_LIBRARIES})
target_link_libraries(sort ${OpenCV_LIBS})
target_link_libraries(yolov5_detect sort)

if(UNIX)
add_definitions(-O2 -pthread)
endif(UNIX)
