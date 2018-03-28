// Standard C++ includes
#include <string>
#include <iostream>
#include <fstream>

// ROS includes
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>

// OpenCV includes
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// Includes for this Library
#include <rgbd_odometry/prob_and_stat_function_dev.h>

void ProbabilityAndStatisticsFunctionProvider::computePoseCovariance(float *x,
        float *y, float *z, float* covMat) {
}