cmake_minimum_required(VERSION 2.8.3)
# this line sets the value of the ${PROJECT_NAME} variable
project(uncc_rgbd_odom_core)

# uncomment this if cmake is not finding your packages
# to view the paths searched and decisions made by cmake
set( CMAKE_FIND_DEBUG_MODE 1 )
#
# Note: CLEAN COMPILE AFTER CHANGING OpenCV_DIR!!!!
#
# HOME PC / Jade
#set( OpenCV_DIR /home/arwillis/opencv-2.4/dist-shared/share/OpenCV )
#set( OpenCV_DIR /home/arwillis/opencv-3.0.0/dist-shared/share/OpenCV )
set ( OpenCV_DIR /opt/ros/kinetic/share/OpenCV-3.3.1)
#set( OpenCV_DIR "/usr/share/OpenCV" )
#
# Work PC / Indigo
#set( OpenCV_DIR /home.old/arwillis/lib/opencv-2.4/dist-shared/share/OpenCV )
#set ( OpenCV_DIR /home.old/arwillis/lib/opencv-3.0/dist-shared/share/OpenCV )
#
set( CMAKE_VERBOSE_MAKEFILE ON )
#set( USE_SYSTEM_VTK OFF)
# Declare variable having additional compiler flags for profiling 
set( GCC_PROFILE_COMPILE_FLAGS "-g -pg" )
set( GCC_PROFILE_LINK_FLAGS "-g -pg" )

#set( GCC_DEFAULT_COMPILE_FLAGS "-std=c++11" )
set( GCC_DEFAULT_COMPILE_FLAGS "-std=gnu++11" )
set( GCC_DEFAULT_LINK_FLAGS "" )

set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GCC_DEFAULT_COMPILE_FLAGS}" )
set( CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${GCC_DEFAULT_LINK_FLAGS}" )

# Add profiling compiler flags for CMAKE_BUILD_TYPE=Debug
set( CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} ${GCC_PROFILE_COMPILE_FLAGS}" )
set( CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS} ${GCC_PROFILE_LINK_FLAGS}" )

# Create a custom build type for CMAKE_BUILD_TYPE=Profile
set( CMAKE_CXX_FLAGS_PROFILE "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG} ${GCC_PROFILE_COMPILE_FLAGS}" )
set( CMAKE_EXE_LINKER_FLAGS_PROFILE "${CMAKE_EXE_LINKER_FLAGS} ${CMAKE_EXE_LINKER_FLAGS_DEBUG} ${GCC_PROFILE_LINK_FLAGS}" )
#set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GCC_PROFILE_COMPILE_FLAGS}" )
#set( CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${GCC_COVERAGE_LINK_FLAGS}" )

## System dependencies are found with CMake's conventions
find_package(Boost REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)

###########
## Build ##
###########

include_directories(
        include
        SYSTEM 
        ${BOOST_INCLUDE_DIRS}
        ${PCL_INCLUDE_DIRS}
        ${EIGEN_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
)

#link_directories(/opt/ros/kinetic/lib)

add_library(${PROJECT_NAME}
                src/rgbd_odometry_core.cpp
                src/RobustMatcher.cpp
                src/image_function_dev.cpp
                src/opencv_function_dev.cpp)

target_link_libraries(${PROJECT_NAME}
    ${PCL_LIBRARIES} 
    ${Boost_LIBRARIES} 
    ${OpenCV_LIBS})
