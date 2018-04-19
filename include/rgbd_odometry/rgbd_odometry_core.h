/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   rgbd_odometry_core.hpp
 * Author: arwillis
 *
 * Created on April 14, 2018, 2:37 PM
 */

#ifndef RGBD_ODOMETRY_CORE_HPP
#define RGBD_ODOMETRY_CORE_HPP

#include <cstdio>
#include <fstream>

#include <boost/shared_ptr.hpp>

// OpenCV includes
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/ocl.hpp>

// OpenCV includes
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// PCL includes
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>

// Eigen includes
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <rgbd_odometry/image_function_dev.h>
#include <rgbd_odometry/RobustMatcher.h>

// Odometry performance data logging variables
static std::string _logfilename("/home/arwillis/odom_transforms_noisy.m");
static std::ofstream fos;

enum Depth_Processing {
    NONE, MOVING_AVERAGE, DITHER
};

class RGBDOdometryCore {
public:
    typedef boost::shared_ptr<RGBDOdometryCore> Ptr;

    RGBDOdometryCore() :
    imageFunctionProvider(new ImageFunctionProvider),
    pcl_ptcloud_sptr(new pcl::PointCloud<pcl::PointXYZRGB>),
    LOG_ODOMETRY_TO_FILE(false),
    //COMPUTE_PTCLOUDS(false),
    DUMP_MATCH_IMAGES(false),
    DUMP_RAW_IMAGES(false),
    SHOW_ORB_vs_iGRaND(false),
    fast_match(false),
    rmatcher(new RobustMatcher()),
    numKeyPoints(600) {
        bool useOpenCL;
        useOpenCL = false;
        std::string opencl_path = ".";
        std::string depthmask_cl = "depthmask.cl";
        getImageFunctionProvider()->initialize(useOpenCL, opencl_path, depthmask_cl);

        std::string feature_detector = "ORB";
        std::string feature_descriptor = "ORB";
        rmatcher->setFeatureDetector(feature_detector);
        rmatcher->setDescriptorExtractor(feature_descriptor);

        std::string depth_processing_str = "none";
        setDepthProcessing(depth_processing_str);
    }

    virtual ~RGBDOdometryCore() {
    }

    bool computeRelativePose(cv::UMat& frame,
            cv::UMat& depthimg,
            Eigen::Matrix4f& trans,
            Eigen::Map<Eigen::Matrix<double, 6, 6> >& covMatrix,
            float& detector_time, float& descriptor_time, float& match_time,
            float& RANSAC_time, float& covarianceTime,
            int& numFeatures, int& numMatches, int& numInliers);

    bool computeRelativePose(cv::UMat &frame, cv::UMat &depthimg,
            Eigen::Matrix4f& trans,
            Eigen::Map<Eigen::Matrix<double, 6, 6> >& covMatrix);

    bool computeRelativePose(cv::UMat &frameA, cv::UMat &depthimgA,
            cv::UMat &frameB, cv::UMat &depthimgB,
            Eigen::Matrix4f& trans,
            Eigen::Map<Eigen::Matrix<double, 6, 6> >& covMatrix);

    int computeKeypointsAndDescriptors(cv::UMat& frame, cv::Mat& dimg, cv::UMat& mask,
            std::string& name,
            cv::Ptr<cv::FeatureDetector> detector_,
            cv::Ptr<cv::DescriptorExtractor> extractor_,
            cv::Ptr<std::vector<cv::KeyPoint> >& keypoints_frame,
            cv::Ptr<cv::UMat>& descriptors_frame, float& detector_time, float& descriptor_time,
            const std::string keyframe_frameid_str);

    bool estimateCovarianceBootstrap(pcl::CorrespondencesPtr ptcloud_matches_ransac,
            cv::Ptr<std::vector<cv::KeyPoint> >& keypoints_frame,
            cv::Ptr<std::vector<cv::KeyPoint> >& prior_keypoints,
            Eigen::Map<Eigen::Matrix<double, 6, 6> >& covMatrix,
            float &covarianceTime);

    ImageFunctionProvider::Ptr getImageFunctionProvider() {
        return imageFunctionProvider;
    }

    bool hasRGBCameraIntrinsics() {
        return rgbCamera_Kmatrix.rows == 3 && rgbCamera_Kmatrix.cols == 3;
    }

    void setRGBCameraIntrinsics(cv::Mat matrix) {
        rgbCamera_Kmatrix = matrix.clone();
    }

    cv::Mat getRGBCameraIntrinsics() {
        return rgbCamera_Kmatrix.clone();
    }

    bool hasMatcher() {
        return rmatcher && rmatcher->detector_ && rmatcher->extractor_;
    }

    void setMatcher(RobustMatcher::Ptr rm) {
        rmatcher = rm;
    }

    RobustMatcher::Ptr getMatcher() {
        return rmatcher;
    }

    void setDepthProcessing(std::string depth_processing_str) {
        if (depth_processing_str.compare("moving_average") == 0) {
            std::cout << "Applying moving average depth filter." << std::endl;
            depth_processing = Depth_Processing::MOVING_AVERAGE;
        } else if (depth_processing_str.compare("dither") == 0) {
            std::cout << "Applying dithering depth filter." << std::endl;
            depth_processing = Depth_Processing::DITHER;
        } else {
            depth_processing = Depth_Processing::NONE;
        }
    }

private:
    // -------------------------
    // Disabling default copy constructor and default
    // assignment operator.
    // -------------------------
    RGBDOdometryCore(const RGBDOdometryCore & yRef);
    RGBDOdometryCore& operator=(const RGBDOdometryCore & yRef);

protected:
    bool LOG_ODOMETRY_TO_FILE;
    //bool COMPUTE_PTCLOUDS;
    bool DUMP_MATCH_IMAGES;
    bool DUMP_RAW_IMAGES;
    bool SHOW_ORB_vs_iGRaND;

    cv::Mat rgbCamera_Kmatrix;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_ptcloud_sptr;

    //std::string prior_keyframe_frameid_str;
    bool fast_match;
    int numKeyPoints;
    RobustMatcher::Ptr rmatcher; // instantiate RobustMatcher

    // class to provide accelerated image processing functions
    ImageFunctionProvider::Ptr imageFunctionProvider;

    // class to provide depth image processing functions
    Depth_Processing depth_processing;


    cv::Ptr<std::vector<cv::KeyPoint> > keypoints_frame;
    cv::Ptr<cv::UMat> descriptors_frame;

    cv::UMat prior_image;
    cv::Ptr<std::vector<cv::KeyPoint> > prior_keypoints;
    cv::Ptr<cv::UMat> prior_descriptors_;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr prior_ptcloud_sptr;
};

#endif /* RGBD_ODOMETRY_CORE_HPP */

