cmake_minimum_required(VERSION 2.8.3)
# this line sets the value of the ${PROJECT_NAME} variable
project(uncc_rgbd_odom)

subdirs(core)
# uncomment this if cmake is not finding your packages
# to view the paths searched and decisions made by cmake
set( CMAKE_FIND_DEBUG_MODE 1 )
#
# Note: CLEAN COMPILE AFTER CHANGING OpenCV_DIR!!!!
#
# HOME PC
#set( OpenCV_DIR /home/arwillis/opencv-2.4/dist-shared/share/OpenCV )
#set( OpenCV_DIR /home/arwillis/opencv-3.0.0/dist-shared/share/OpenCV )
#set( OpenCV_DIR "/opt/ros/indigo/share/OpenCV-3.1.0-dev")
#
#
# ODROID
#set( OpenCV_DIR /usr/share/OpenCV )
#set( OpenCV_DIR /usr/local/share/OpenCV )
#set( OpenCV_DIR /home/odroid/opencv-2.4.9/build )
set( CMAKE_VERBOSE_MAKEFILE ON )
set( USE_SYSTEM_VTK OFF)

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
## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED roscpp rospy genmsg std_msgs geometry_msgs sensor_msgs
             cmake_modules eigen_conversions 
             cv_bridge image_geometry image_proc image_transport nodelet 
# freenect_launch not available for jade on ubuntu yet 1/4/2016
#             tf2 tf2_ros pcl_ros freenect_launch openni2_launch rgbd_launch)
#             tf2 tf2_ros pcl_ros openni2_launch rgbd_launch rgbd_drivers_uncc)
             tf2 tf2_ros pcl_ros rgbd_drivers_uncc)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)
find_package(cmake_modules REQUIRED)
find_package(Boost REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
# catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################

## To declare and build messages, services or actions from within this
## package, follow these steps:
## * Let MSG_DEP_SET be the set of packages whose message types you use in
##   your messages/services/actions (e.g. std_msgs, actionlib_msgs, ...).
## * In the file package.xml:
##   * add a build_depend and a run_depend tag for each package in MSG_DEP_SET
##   * If MSG_DEP_SET isn't empty the following dependencies might have been
##     pulled in transitively but can be declared for certainty nonetheless:
##     * add a build_depend tag for "message_generation"
##     * add a run_depend tag for "message_runtime"
## * In this file (CMakeLists.txt):
##   * add "message_generation" and every package in MSG_DEP_SET to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * add "message_runtime" and every package in MSG_DEP_SET to
##     catkin_package(CATKIN_DEPENDS ...)
##   * uncomment the add_*_files sections below as needed
##     and list every .msg/.srv/.action file to be processed
##   * uncomment the generate_messages entry below
##   * add every package in MSG_DEP_SET to generate_messages(DEPENDENCIES ...)

## Generate messages in the 'msg' folder
# add_message_files(
#   FILES
#   Message1.msg
#   Message2.msg
# )

## Generate services in the 'srv' folder
# add_service_files(
#   FILES
#   Service1.srv
#   Service2.srv
# )

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
# generate_messages(
#   DEPENDENCIES
#   std_msgs  # Or other packages containing msgs
# )
#generate_messages(DEPENDENCIES std_msgs geometry_msgs sensor_msgs nav_msgs)
#generate_dynamic_reconfigure_options(
#  cfg/nodeExample.cfg
#)

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if you package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES relative_nav_msgs_subscriber
#  CATKIN_DEPENDS other_catkin_pkg
#CATKIN_DEPENDS message_runtime
    # the INCLUDE below allows us to include our headers 
    # using #include <${PROJECT_NAME}/XXXX.h> by creating a 
    # folder having name include/${PROJECT_NAME} from the package root folder
    INCLUDE_DIRS include core/include
    LIBRARIES uncc_rgbd_odom_core
#    LIBRARIES ${PROJECT_NAME}_core
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
# include_directories(include)

## Declare a cpp library
# add_library(relative_nav_msgs_subscriber
#   src/${PROJECT_NAME}/relative_nav_msgs_subscriber.cpp
# )

## Declare a cpp executable
# add_executable(relative_nav_msgs_subscriber_node src/relative_nav_msgs_subscriber_node.cpp)
#include_directories(~/lib/opencv-2.4/dist/include include ${catkin_INCLUDE_DIRS})
#include_directories(include ${catkin_INCLUDE_DIRS})
include_directories(
        include
        core/include
        SYSTEM 
        ${BOOST_INCLUDE_DIRS}
        ${catkin_INCLUDE_DIRS}
        ${EIGEN_INCLUDE_DIRS}
        ${PCL_INCLUDE_DIRS}
)

link_directories(/opt/ros/kinetic/lib
    /home/arwillis/astra/OpenNI-Linux-x64-2.3/Redist) # needed to find OpenNI2Orbbec

#add_library(${PROJECT_NAME}
#                src/rgbd_odometry_core.cpp
#                src/RobustMatcher.cpp
#                src/image_function_dev.cpp
#                src/opencv_function_dev.cpp)

#target_link_libraries(${PROJECT_NAME}
#    ${catkin_LIBRARIES} 
#    ${Boost_LIBRARIES} 
#    ${OpenCV_LIBS})

add_executable(${PROJECT_NAME}
#                src/rgbd_odometry_core.cpp
                src/rgbd_odometry_ros.cpp)
#                src/RobustMatcher.cpp
#                src/image_function_dev.cpp
#                src/opencv_function_dev.cpp)

target_link_libraries(${PROJECT_NAME}
    ${catkin_LIBRARIES} 
    ${Boost_LIBRARIES} 
    ${OpenCV_LIBS}
#    ${PROJECT_NAME})
    uncc_rgbd_odom_core)
#target_link_libraries(feature3d_engine ${catkin_LIBRARIES} opencv_nonfree)
#add_dependencies(rgbd_odometry geometry_msgs_gencpp nav_msgs_gencpp)
#add_dependencies(${PROJECT_NAME} image_geometry)
## Add cmake target dependencies of the executable/library
## as an example, message headers may need to be generated before nodes
# add_dependencies(relative_nav_msgs_subscriber_node relative_nav_msgs_subscriber_generate_messages_cpp)
## Specify libraries to link a library or executable target against
# target_link_libraries(relative_nav_msgs_subscriber_node
#   ${catkin_LIBRARIES}
# )

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# install(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )
#install(TARGETS ${PROJECT_NAME}
#        DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#)

install(DIRECTORY include/${PROJECT_NAME}/
        DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)
## Mark executables and/or libraries for installation
# install(TARGETS relative_nav_msgs_subscriber relative_nav_msgs_subscriber_node
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_relative_nav_msgs_subscriber.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
