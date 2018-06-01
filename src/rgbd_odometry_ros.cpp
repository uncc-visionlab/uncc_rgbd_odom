/*
 * File:   Feature3DEngine.cpp
 * Author: arwillis
 *
 * Created on August 18, 2015, 10:35 AM
 */
#include <math.h>
#include <iostream>
#include <fstream>
#include <string>
#include <iterator>
#include <random>

// ROS includes
#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <message_filters/subscriber.h>
//#include <message_filters/time_synchronizer.h>
//#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>
#include <pcl_conversions/pcl_conversions.h>
#include <eigen_conversions/eigen_msg.h>

// OpenCV includes
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// Eigen includes
#include <Eigen/Dense>
#include <Eigen/Geometry>

// PCL includes
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>

// TF includes
#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>

// Includes for this Library
#include <rgbd_odometry/rgbd_odometry_ros.h>
#include <cv_bridge/cv_bridge.h>

void toString(pcl::PointXYZRGB& ptrgb) {
    ROS_INFO("x,y,z=(%f,%f,%f) r,g,b=(%d,%d,%d)",
            ptrgb.x, ptrgb.y, ptrgb.z,
            ptrgb.rgba >> 24, (ptrgb.rgba & 0x00FFFFFF) >> 16, (ptrgb.rgba & 0x0000FFFF) >> 8);
}

void logInitialTransformData(std::string frameid, ros::Time frame_time,
        Eigen::Quaternionf quat, Eigen::Vector3f trans,
        Eigen::Matrix4f transform) {
    static int StreamPrecision = 8;
    static Eigen::IOFormat OctaveFmt(StreamPrecision, 0, ", ", ";\n", "", "", "[", "]");
    if (!fos) {
        std::cout << "Opening logfile " << _logfilename << "." << std::endl;
        fos.open(_logfilename.c_str());
    }
    fos << "rgbd_odometry_init.frame_id = '" << frameid << "';" << std::endl;
    fos << "rgbd_odometry_init.sec = " << frame_time.sec << ";" << std::endl;
    fos << "rgbd_odometry_init.nsec = " << frame_time.nsec << ";" << std::endl;
    fos << "rgbd_odometry_init.quaternion = [";
    fos << quat.x() << " " << quat.y() << " " << quat.z() << " " << quat.w() << "];" << std::endl;
    fos << "rgbd_odometry_init.translation = [";
    fos << trans.x() << " " << trans.y() << " " << trans.z() << "];" << std::endl;
    fos << "rgbd_odometry_init.transform = ";
    fos << transform.format(OctaveFmt) << ";" << std::endl;
}

namespace Eigen {

    void toString(std::string name, Eigen::MatrixXf mat) {
        static std::string sep = "\n----------------------------------------\n";
        static int StreamPrecision = 4;
        static Eigen::IOFormat OctaveFmt(StreamPrecision, 0, ", ", ";\n", "", "", "[", "]");
        std::cout << sep << name << " = " << mat.format(OctaveFmt) << ";" << sep;
    }
}

void RGBDOdometryEngine::tf_truth_Callback(const geometry_msgs::TransformStampedConstPtr& tf_truth) {
    // UNCOMMENT WHEN USING vrpn_client_ros 
    //void RGBDOdometryEngine::tf_truth_Callback(const geometry_msgs::PoseStampedConstPtr& tf_truth) {
    //    static geometry_msgs::PoseStampedConstPtr first_tf;
    static geometry_msgs::TransformStampedConstPtr first_tf;
    if (!first_tf) {
        first_tf = tf_truth;
    }

    static tf::TransformListener listener;
    tf::StampedTransform calib_marker_pose;
    if (VERBOSE) {
        ROS_INFO("Looking up transform from frame '%s' to frame '%s'", parent_frame_id_str.c_str(),
                calib_frame_id_str.c_str());
    }
    //ros::Time queryTime(ros::Time(0));
    ros::Time queryTime = ros::Time::now();
    //ros::Time queryTime(ros::Time::now()-ros::Duration(0.1));
    try {
        listener.waitForTransform(parent_frame_id_str, calib_frame_id_str,
                queryTime, ros::Duration(1));
        listener.lookupTransform(parent_frame_id_str, calib_frame_id_str,
                queryTime, calib_marker_pose);
    } catch (tf::TransformException ex) {
        ROS_ERROR("%s", ex.what());
        //        ros::Duration(1.0).sleep();
    }

    //    geometry_msgs::Vector3 translation;
    //    geometry_msgs::Quaternion quaternion;
    tf::Vector3 translation;
    tf::Quaternion quaternion;
    translation = calib_marker_pose.getOrigin();
    quaternion = calib_marker_pose.getRotation();

    tf::Vector3 tval(translation.x(), translation.y(), translation.z());
    tf::Quaternion qval(quaternion.x(), quaternion.y(),
            quaternion.z(), quaternion.w());
    qval = qval.normalize();
    tf::Transform initialTransform(qval, tval);
    setInitialTransform(initialTransform, false);
    br.sendTransform(tf::StampedTransform(rgbd_pose,
            frame_time, parent_frame_id_str, rgbd_frame_id_str));

    ROS_INFO("Heard tf_truth.");
    //    std::cout << "rotation = " << tf_truth->transform.rotation << std::endl
    //            << "translation = " << tf_truth->transform.translation << std::endl;
    if (tf_truth->header.stamp.sec - first_tf->header.stamp.sec > tf_truth_init_time) {
        initializationDone = true;
        std::cout << "Initialization time has expired." << std::endl;
        std::cout << " Setting ground truth transform to: " << std::endl
                << "rotation = (" << qval.getX() << ", " << qval.getY()
                << ", " << qval.getZ() << ", " << qval.getW() << ")" << std::endl
                << "translation = (" << tval.getX() << ", " << tval.getY()
                << ", " << tval.getZ() << ")" << std::endl;
        sub_tf_truth.shutdown();
    }
}

void RGBDOdometryEngine::setInitialTransform(tf::Transform target_pose, bool isOdomPose) {
    if (isOdomPose) {
        // switch to optical frame (X,Y,Z) = (right,down,forward)
        // from odom frame (X,Y,Z) = (forward, left, up)
        tf::Quaternion nav_frame = target_pose.getRotation();
        tf::Matrix3x3 rotation_baselink2optical;
        rotation_baselink2optical.setRPY(-M_PI / 2, 0, -M_PI / 2);
        tf::Vector3 trans;
        trans = target_pose.getOrigin();
        tf::Quaternion opt_frame;
        rotation_baselink2optical.getRotation(opt_frame);
        nav_frame *= opt_frame;
        tf::Transform opt_pose(nav_frame, trans);
        rgbd_pose = opt_pose;
    } else {
        rgbd_pose = target_pose;
    }
    if (LOG_ODOMETRY_TO_FILE) {
        tf::Vector3 tf_tran = rgbd_pose.getOrigin();
        tf::Quaternion tf_quat = rgbd_pose.getRotation();
        Eigen::Vector3f e_trans;
        e_trans.x() = tf_tran.getX();
        e_trans.y() = tf_tran.getY();
        e_trans.z() = tf_tran.getZ();
        Eigen::Quaternionf e_quat;
        e_quat.x() = tf_quat.getX();
        e_quat.y() = tf_quat.getY();
        e_quat.z() = tf_quat.getZ();
        e_quat.w() = tf_quat.getW();
        Eigen::Matrix4f e_transform = Eigen::Matrix4f::Zero();
        for (int rowidx = 0; rowidx < 3; rowidx++) {
            tf::Vector3 rowvals = target_pose.getBasis().getRow(rowidx);
            e_transform(rowidx, 0) = rowvals.getX();
            e_transform(rowidx, 1) = rowvals.getY();
            e_transform(rowidx, 2) = rowvals.getZ();
        }
        e_transform(0, 3) = e_trans.x();
        e_transform(1, 3) = e_trans.y();
        e_transform(2, 3) = e_trans.z();
        e_transform(3, 3) = 1;
        logInitialTransformData("initial", ros::Time::now(),
                e_quat, e_trans, e_transform);
    }
}

void RGBDOdometryEngine::changePose(tf::Transform xform) {
    tf::Transform new_pose;
    //        tf::Quaternion delta_frame = xform.getRotation();
    //        tf::Vector3 delta_origin = xform.getOrigin();
    //        tf::Quaternion optical_frame = rgbd_pose.getRotation();
    //        tf::Vector3 optical_origin = rgbd_pose.getOrigin();
    //        optical_frame *= delta_frame;
    //        optical_origin += delta_origin;
    //        new_pose.setRotation(optical_frame);
    //        new_pose.setOrigin(optical_origin);
    new_pose.mult(rgbd_pose, xform);
    rgbd_pose = new_pose;

    //        geometry_msgs::Transform test2;
    //        tf::transformTFToMsg(test, test2);
    //        std::cout << "rgbd_pose = " << test2 << std::endl;

    br.sendTransform(tf::StampedTransform(rgbd_pose,
            frame_time, parent_frame_id_str, rgbd_frame_id_str));
}

std::pair<cv::Mat, cv::Mat> RGBDOdometryEngine::cameraInfoToMats(const sensor_msgs::CameraInfoConstPtr& camera_info, bool rectified) {

    cv::Mat camera_matrix(3, 3, CV_32FC1);
    cv::Mat distortion_coeffs(5, 1, CV_32FC1);

    if (rectified) {

        camera_matrix.at<float>(0, 0) = camera_info->P[0];
        camera_matrix.at<float>(0, 1) = camera_info->P[1];
        camera_matrix.at<float>(0, 2) = camera_info->P[2];
        camera_matrix.at<float>(1, 0) = camera_info->P[4];
        camera_matrix.at<float>(1, 1) = camera_info->P[5];
        camera_matrix.at<float>(1, 2) = camera_info->P[6];
        camera_matrix.at<float>(2, 0) = camera_info->P[8];
        camera_matrix.at<float>(2, 1) = camera_info->P[9];
        camera_matrix.at<float>(2, 2) = camera_info->P[10];

        for (int i = 0; i < 5; ++i)
            distortion_coeffs.at<float>(i, 0) = 0;

    } else {

        for (int i = 0; i < 9; ++i) {
            camera_matrix.at<float>(i / 3, i % 3) = camera_info->K[i];
        }
        for (int i = 0; i < 5; ++i) {
            distortion_coeffs.at<float>(i, 0) = camera_info->D[i];
        }

    }

    return std::make_pair(camera_matrix, distortion_coeffs);

}

void RGBDOdometryEngine::rgbdCallback(const sensor_msgs::ImageConstPtr& depth_msg,
        const sensor_msgs::ImageConstPtr& rgb_msg,
        const sensor_msgs::CameraInfoConstPtr& info_msg) {

    ros::Time timestamp = depth_msg->header.stamp;
    static int frame_id = 0;
    if (VERBOSE) {
        ROS_DEBUG("Heard rgbd image.");
    }
    frame_time = depth_msg->header.stamp;
    std::string keyframe_frameid_str("frame_");
    keyframe_frameid_str.append(stdpatch::to_string(frame_id++));

    cv_bridge::CvImageConstPtr rgb_img_ptr = cv_bridge::toCvShare(rgb_msg, sensor_msgs::image_encodings::BGR8);
    cv_bridge::CvImageConstPtr depth_img_ptr;
    if (depth_msg->encoding == sensor_msgs::image_encodings::TYPE_16UC1) {
        depth_img_ptr = cv_bridge::toCvShare(depth_msg, sensor_msgs::image_encodings::TYPE_16UC1);
    } else if (depth_msg->encoding == sensor_msgs::image_encodings::TYPE_32FC1) {
        depth_img_ptr = cv_bridge::toCvShare(depth_msg, sensor_msgs::image_encodings::TYPE_32FC1);
    }
    if (depth_msg->width != rgb_msg->width || depth_msg->height != rgb_msg->height) {
        ROS_ERROR("Depth and RGB image dimensions don't match depth=( %dx%d ) rgb=( %dx%d )!",
                depth_msg->width, depth_msg->height, rgb_msg->width, rgb_msg->height);
    }

    if (!hasRGBCameraIntrinsics()) {
        cv::Mat cameraMatrix, distortionCoeffs;
        std::tie(cameraMatrix, distortionCoeffs) = cameraInfoToMats(info_msg, false);

        float cx = cameraMatrix.at<float>(0, 2);
        float cy = cameraMatrix.at<float>(1, 2);
        float fx = cameraMatrix.at<float>(0, 0);
        float fy = cameraMatrix.at<float>(1, 1);
        setRGBCameraIntrinsics(cameraMatrix);
    }

    Eigen::Matrix4f trans;
    double cov[36];
    Eigen::Map<Eigen::Matrix<double, 6, 6> > covMatrix(cov);
    cv::UMat depthimg = depth_img_ptr->image.getUMat(cv::ACCESS_READ);
    cv::UMat frame = rgb_img_ptr->image.getUMat(cv::ACCESS_READ);
    bool odomEstimatorSuccess = computeRelativePose(frame, depthimg, trans, covMatrix);

    Eigen::Quaternionf quat(trans.block<3, 3>(0, 0));
    Eigen::Vector3f translation(trans.block<3, 1>(0, 3));

    if (initializationDone && odomEstimatorSuccess) {
        tf::Quaternion tf_quat(quat.x(), quat.y(), quat.z(), quat.w());
        tf::Transform xform(tf_quat,
                tf::Vector3(translation[0], translation[1], translation[2]));
        tf::StampedTransform xformStamped(xform, frame_time, keyframe_frameid_str, keyframe_frameid_str);
        geometry_msgs::TransformStamped gxform;
        tf::transformStampedTFToMsg(xformStamped, gxform);
        gxform.header.frame_id = keyframe_frameid_str;
        // publish geometry_msgs::pose with covariance message
        geometry_msgs::PoseWithCovarianceStamped pose_w_cov_msg;
        geometry_msgs::PoseWithCovarianceStamped odom_w_cov_msg;
        geometry_msgs::TransformStamped pose_transform;
        tf::Transform new_pose;
        new_pose.mult(rgbd_pose, xform);
        tf::StampedTransform new_pose_stamped(new_pose, frame_time, "pose", "");
        tf::transformStampedTFToMsg(new_pose_stamped, pose_transform);
        pose_w_cov_msg.pose.pose.orientation = pose_transform.transform.rotation;
        pose_w_cov_msg.pose.pose.position.x = pose_transform.transform.translation.x;
        pose_w_cov_msg.pose.pose.position.y = pose_transform.transform.translation.y;
        pose_w_cov_msg.pose.pose.position.z = pose_transform.transform.translation.z;
        for (int offset = 0; offset < 36; offset++) {
            pose_w_cov_msg.pose.covariance[offset] = cov[offset];
        }
        pose_w_cov_msg.header.stamp = frame_time;
        pose_w_cov_msg.header.frame_id = keyframe_frameid_str;
        pubPose_w_cov.publish(pose_w_cov_msg);

        odom_w_cov_msg.pose.pose.orientation = gxform.transform.rotation;
        odom_w_cov_msg.pose.pose.position.x = gxform.transform.translation.x;
        odom_w_cov_msg.pose.pose.position.y = gxform.transform.translation.y;
        odom_w_cov_msg.pose.pose.position.z = gxform.transform.translation.z;
        for (int offset = 0; offset < 36; offset++) {
            odom_w_cov_msg.pose.covariance[offset] = cov[offset];
        }
        odom_w_cov_msg.header.stamp = frame_time;
        odom_w_cov_msg.header.frame_id = keyframe_frameid_str;
        pubOdom_w_cov.publish(odom_w_cov_msg);

        // publish current estimated pose to tf and update current pose estimate
        changePose(xform);

        // publish geometry_msgs::StampedTransform message
        pubXforms.publish(gxform);
    }
    prior_keyframe_frameid_str = keyframe_frameid_str;
}

void RGBDOdometryEngine::tofRGBImageCallback(const sensor_msgs::ImageConstPtr& x_msg,
        const sensor_msgs::ImageConstPtr& y_msg,
        const sensor_msgs::ImageConstPtr& z_msg,
        const sensor_msgs::ImageConstPtr& rgb_msg,
        const sensor_msgs::ImageConstPtr& uv) {

}

void RGBDOdometryEngine::tofGreyImageCallback(const sensor_msgs::ImageConstPtr& depth_img,
        const sensor_msgs::ImageConstPtr& rgb_msg_in,
        const sensor_msgs::CameraInfoConstPtr& info_msg) {

}

// Detectors/Descriptors: ORB, SIFT, SURF, BRISK
//  Detector-only algorithms: FAST, GFTT
//  Descriptor-only algorithms: BRIEF 
#define NUM_TESTS 1
int trackedIdx = 0;
#ifdef PERFORMANCE_EVAL
std::string odomAlgorithmPairs[NUM_TESTS][2] = {
    {"iGRAND", "iGRAND"},
    {"ORB", "ORB"}
    //    {"SIFT", "SIFT"},
    //    {"SURF", "SURF"},
    //    {"BRISK", "BRISK"},
    //    {"ORB", "SIFT"},
    //    {"ORB", "SURF"},
    //{"GFTT", "SIFT"}
};
std::vector< cv::Ptr<std::vector<cv::KeyPoint> > > vec_prior_keypoints(NUM_TESTS);
std::vector< cv::Ptr<cv::UMat> > vec_prior_descriptors_(NUM_TESTS);
std::vector< pcl::PointCloud<pcl::PointXYZRGB>::Ptr > vec_prior_ptcloud_sptr_(NUM_TESTS);
#endif

void RGBDOdometryEngine::rgbdImageCallback(const sensor_msgs::ImageConstPtr& depth_msg,
        const sensor_msgs::ImageConstPtr& rgb_msg_in,
        const sensor_msgs::CameraInfoConstPtr& info_msg) {
    static int frame_id = 0;
    if (VERBOSE) {
        ROS_DEBUG("Heard rgbd image.");
    }
    frame_time = depth_msg->header.stamp;
    std::string keyframe_frameid_str("frame_");
    keyframe_frameid_str.append(stdpatch::to_string(frame_id++));
    depth_encoding = depth_msg->encoding;
    cv_rgbimg_ptr = cv_bridge::toCvShare(rgb_msg_in, sensor_msgs::image_encodings::BGR8);
    if (depth_msg->encoding == sensor_msgs::image_encodings::TYPE_16UC1) {
        cv_depthimg_ptr = cv_bridge::toCvShare(depth_msg, sensor_msgs::image_encodings::TYPE_16UC1);
    } else if (depth_msg->encoding == sensor_msgs::image_encodings::TYPE_32FC1) {
        cv_depthimg_ptr = cv_bridge::toCvShare(depth_msg, sensor_msgs::image_encodings::TYPE_32FC1);
    }
    if (depth_msg->width != rgb_msg_in->width || depth_msg->height != rgb_msg_in->height) {
        ROS_ERROR("Depth and RGB image dimensions don't match depth=( %fx%f ) rgb=( %dx%d )!",
                depth_msg->width, depth_msg->height, rgb_msg_in->width, rgb_msg_in->height);
    }
    model_.fromCameraInfo(info_msg);
    frame_id_str = keyframe_frameid_str;
    if (VERBOSE) {
        ROS_DEBUG("Computing relative pose for frame id %d.", frame_id);
    }

    // pointer to the feature point detector object
    //cv::Ptr<cv::FeatureDetector> detector_;
    // pointer to the feature descriptor extractor object
    //cv::Ptr<cv::DescriptorExtractor> extractor_;
    cv::UMat depthimg = cv_depthimg_ptr->image.getUMat(cv::ACCESS_READ);
    cv::UMat frame = cv_rgbimg_ptr->image.getUMat(cv::ACCESS_READ);

    bool odomEstimatorSuccess;
    float detectorTime, descriptorTime, matchTime, RANSACTime, covarianceTime;
    for (int pairIdx = 0; pairIdx < NUM_TESTS; pairIdx++) {
#ifdef PERFORMANCE_EVAL
        std::string detectorStr = odomAlgorithmPairs[pairIdx][0];
        std::string descriptorStr = odomAlgorithmPairs[pairIdx][1];
        rmatcher->setFeatureDetector(detectorStr);
        rmatcher->setDescriptorExtractor(descriptorStr);

        prior_keypoints = vec_prior_keypoints[pairIdx];
        prior_descriptors_ = vec_prior_descriptors_[pairIdx];
        prior_ptcloud_sptr = vec_prior_ptcloud_sptr_[pairIdx];
        cv::Ptr<std::vector<cv::KeyPoint> > keypoints_frame(new std::vector<cv::KeyPoint>);
        cv::Ptr<cv::UMat> descriptors_frame(new cv::UMat);
#endif
        double cov[36];
        Eigen::Map<Eigen::Matrix<double, 6, 6 >> covMatrix(cov);
        Eigen::Matrix4f trans;
        int numFeatures = 0, numMatches = 0, numInliers = 0;

        //std::cout << "Detector = " << detectorStr << " Descriptor = " << descriptorStr << std::endl;
        odomEstimatorSuccess = computeRelativePose(frame, depthimg,
                trans, covMatrix,
                detectorTime, descriptorTime, matchTime, RANSACTime, covarianceTime,
                numFeatures, numMatches, numInliers);
        std::cout << "trans=\n" << trans << std::endl;
        if (!odomEstimatorSuccess) {
            return;
        }
        Eigen::Quaternionf quat(trans.block<3, 3>(0, 0));
        Eigen::Vector3f translation(trans.block<3, 1>(0, 3));
        //        if (LOG_ODOMETRY_TO_FILE) {
        //            logTransformData(keyframe_frameid_str, frame_time,
        //                    rmatcher->detectorStr, rmatcher->descriptorStr,
        //                    detectorTime, descriptorTime, matchTime, RANSACTime, covarianceTime,
        //                    numFeatures, numMatches, numInliers,
        //                    quat, translation,
        //                    trans, covMatrix, transform_vector);
        //        }
        if (pairIdx == trackedIdx && initializationDone) {
            tf::Quaternion tf_quat(quat.x(), quat.y(), quat.z(), quat.w());
            tf::Transform xform(tf_quat,
                    tf::Vector3(translation[0], translation[1], translation[2]));
            tf::StampedTransform xformStamped(xform, frame_time, keyframe_frameid_str, keyframe_frameid_str);
            geometry_msgs::TransformStamped gxform;
            tf::transformStampedTFToMsg(xformStamped, gxform);
            gxform.header.frame_id = keyframe_frameid_str;
            // publish geometry_msgs::pose with covariance message
            geometry_msgs::PoseWithCovarianceStamped pose_w_cov_msg;
            geometry_msgs::PoseWithCovarianceStamped odom_w_cov_msg;
            geometry_msgs::TransformStamped pose_transform;
            tf::Transform new_pose;
            new_pose.mult(rgbd_pose, xform);
            tf::StampedTransform new_pose_stamped(new_pose, frame_time, "pose", "");
            tf::transformStampedTFToMsg(new_pose_stamped, pose_transform);
            pose_w_cov_msg.pose.pose.orientation = pose_transform.transform.rotation;
            pose_w_cov_msg.pose.pose.position.x = pose_transform.transform.translation.x;
            pose_w_cov_msg.pose.pose.position.y = pose_transform.transform.translation.y;
            pose_w_cov_msg.pose.pose.position.z = pose_transform.transform.translation.z;
            for (int offset = 0; offset < 36; offset++) {
                pose_w_cov_msg.pose.covariance[offset] = cov[offset];
            }
            pose_w_cov_msg.header.stamp = frame_time;
            pose_w_cov_msg.header.frame_id = keyframe_frameid_str;
            pubPose_w_cov.publish(pose_w_cov_msg);

            odom_w_cov_msg.pose.pose.orientation = gxform.transform.rotation;
            odom_w_cov_msg.pose.pose.position.x = gxform.transform.translation.x;
            odom_w_cov_msg.pose.pose.position.y = gxform.transform.translation.y;
            odom_w_cov_msg.pose.pose.position.z = gxform.transform.translation.z;
            for (int offset = 0; offset < 36; offset++) {
                odom_w_cov_msg.pose.covariance[offset] = cov[offset];
            }
            odom_w_cov_msg.header.stamp = frame_time;
            odom_w_cov_msg.header.frame_id = keyframe_frameid_str;
            pubOdom_w_cov.publish(odom_w_cov_msg);

            // publish current estimated pose to tf and update current pose estimate
            changePose(xform);

            // publish geometry_msgs::StampedTransform message
            pubXforms.publish(gxform);
        }
    }
    prior_keyframe_frameid_str = keyframe_frameid_str;
}

void RGBDOdometryEngine::initializeSubscribersAndPublishers() {
    int queue_size = 10;
    //ros::NodeHandlePtr nodeptr(new ros::NodeHandle);
    //nodeptr(new ros::NodeHandle)
    image_transport::ImageTransport it_depth(*nodeptr);
    // parameter for depth_image_transport hint
    std::string depth_image_transport_param = "depth_image_transport";
    // depth image can use different transport.(e.g. compressedDepth)
    image_transport::TransportHints depth_hints("raw", ros::TransportHints(),
            *nodeptr, depth_image_transport_param);
    //    image_transport::Subscriber sub_depthImage = it_depth.subscribe("depth/image_raw", 1, depth_hints);
    //image_transport::SubscriberFilter sub_depthImage;
    sub_depthImage.subscribe(it_depth, "depth_registered/input_image", 1, depth_hints);

    //message_filters::Subscriber<sensor_msgs::CameraInfo>
    //        sub_rgbImage(*nodeptr, "rgb/image_raw", 1);
    image_transport::ImageTransport it_rgb(*nodeptr);
    // rgb uses normal ros transport hints.
    image_transport::TransportHints hints("raw", ros::TransportHints(), *nodeptr);
    //image_transport::SubscriberFilter sub_rgbImage;
    sub_rgbImage.subscribe(it_rgb, "rgb/input_image", 1, hints);

    //    message_filters::Subscriber<sensor_msgs::CameraInfo>
    //            sub_rgbCameraInfo;
    sub_rgbCameraInfo.subscribe(*nodeptr, "rgb/camera_info", 1);

    // option 1
    //    message_filters::TimeSynchronizer
    //            <sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::CameraInfo>
    //            syncTime(sub_depthImage, sub_rgbImage, sub_rgbCameraInfo, queue_size);
    //    syncTime.registerCallback(boost::bind(&Feature3DEngine::rgbdImageCallback,
    //            engineptr, _1, _2, _3));
    // option 2
    //    typedef message_filters::sync_policies::ExactTime
    //            <sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::CameraInfo> MyExactSyncPolicy;
    // ExactTime takes a queue size as its constructor argument, hence MySyncPolicy(queue_size)
    //    message_filters::Synchronizer<MyExactSyncPolicy> syncExact(MyExactSyncPolicy(queue_size),
    //            sub_depthImage, sub_rgbImage, sub_rgbCameraInfo);
    // option 3
    typedef message_filters::sync_policies::ApproximateTime
            <sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::CameraInfo> MyApproximateSyncPolicy;
    static message_filters::Synchronizer<MyApproximateSyncPolicy> syncApprox(MyApproximateSyncPolicy(queue_size),
            sub_depthImage, sub_rgbImage, sub_rgbCameraInfo);
    //syncApprox.registerCallback(boost::bind(&RGBDOdometryEngine::rgbdImageCallback,
    //        this, _1, _2, _3));
    syncApprox.registerCallback(boost::bind(&RGBDOdometryEngine::rgbdCallback,
            this, _1, _2, _3));

    pubXforms = nodeptr->advertise<geometry_msgs::TransformStamped>("relative_xform", 1000);
    pubPose_w_cov = nodeptr->advertise<geometry_msgs::PoseWithCovarianceStamped>("pose_w_cov", 1000);
    pubOdom_w_cov = nodeptr->advertise<geometry_msgs::PoseWithCovarianceStamped>("odom_w_cov", 1000);
}

int main(int argc, char **argv) {
    /**
     * The ros::init() function needs to see argc and argv so that it can perform
     * any ROS arguments and name remapping that were provided at the command line.
     * For programmatic remappings you can use a different version of init() which takes
     * remappings directly, but for most command-line programs, passing argc and argv is
     * the easiest way to do it.  The third argument to init() is the name of the node.
     *
     * You must call one of the versions of ros::init() before using any other
     * part of the ROS system.
     */
    ros::init(argc, argv, "uncc_rgbd_odom");
    /**
     * NodeHandle is the main access point to communications with the ROS system.
     * The first NodeHandle constructed will fully initialize this node, and the last
     * NodeHandle destructed will close down the node.
     */
    RGBDOdometryEngine engine;

    // %Tag(SUBSCRIBER)%
    engine.initializeSubscribersAndPublishers();

    engine.getImageFunctionProvider()->computeFilterBank();
    /**
     * ros::spin() will enter a loop, pumping callbacks.  With this version, all
     * callbacks will be called from within this thread (the main one).  ros::spin()
     * will exit when Ctrl-C is pressed, or the node is shutdown by the master.
     */
    ros::spin();
    /*
    int rateVal = 5;
    ros::Rate rate(rateVal);
    int i = 0;
    while (true) {
        ros::spinOnce();
        rate.sleep();
    }
     */

    engine.getImageFunctionProvider()->freeFilterBank();
    return 0;
}



