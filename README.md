# uncc_rgbd_odom
This package performs odometry estimation for an RGBD sensor.

Non-ROS Build
====================================================================
You can build the base odometry library (libuncc_rgbd_odom_core.so) by using
cmake from inside the "core" folder.

The base classes for odometry depend upon the following libraries:

OpenCV
PCL
Eigen

An example build session is below:

$> cd core
$> mkdir build
$> cd build
$> cmake ..
$> make

The product is the library file: libuncc_rgbd_odom_core.a

You can then link against this file to use core UNCC odometry code. The odometry
algorithm API can be found in include/rgbd_odometry/rgbd_odometry.h.

ROS Build
====================================================================
You can also include the parent uncc_rgbd_odom folder as a standard ROS package. In
this case, the package.xml and CMakeLists.txt in the source root folder control the build
process for the ROS package. The dependencies are listed in the CMakeLists.txt file.


