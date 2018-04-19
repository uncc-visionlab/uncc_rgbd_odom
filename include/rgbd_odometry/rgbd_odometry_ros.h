/* 
 * File:   Feature3DEngine.h
 * Author: arwillis
 *
 * Created on August 18, 2015, 10:35 AM
 */

#ifndef RGBD_ODOMETRY_H
#define RGBD_ODOMETRY_H

// Standard C++ includes
#include <string>
#include <iostream>
#include <math.h>

// ROS includes
#include <ros/ros.h>
#include <message_filters/subscriber.h>
#include <std_msgs/String.h>
#include <std_msgs/Header.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/image_encodings.h>
#include <geometry_msgs/TransformStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>

#include <image_transport/subscriber_filter.h>
#include <image_geometry/pinhole_camera_model.h>
#include <visualization_msgs/Marker.h>
#include <cv_bridge/cv_bridge.h>
#include <tf/tf.h>
#include <tf/transform_broadcaster.h>

#include <rgbd_odometry/rgbd_odometry_core.h>

#define NUMIDCHARS 3
#define DEBUG false
//#define PERFORMANCE_EVAL false

namespace stdpatch {

    template < typename T > std::string to_string(const T& n) {
        std::ostringstream stm;
        stm << std::setw(NUMIDCHARS) << std::setfill('0') << n;
        //stm << n;
        return stm.str();
    }
}

class RGBDOdometryEngine : public RGBDOdometryCore {
public:
    typedef boost::shared_ptr<RGBDOdometryEngine> Ptr;

    RGBDOdometryEngine() : 
    RGBDOdometryCore(),
    nodeptr(new ros::NodeHandle),
    nh("~"),
    initializationDone(false) {
        std::string opencl_path, depthmask_cl, tf_truth_topic, calibration_pose;
        std::string optical_parent, optical_frame, depth_processing_str;
        std::string feature_detector, feature_descriptor;
        bool useOpenCL, tf_truth_initialize;

        nh.param<std::string>("OpenCL_path", opencl_path, ".");
        nh.param<std::string>("depthmask_cl", depthmask_cl, "depthmask.cl");
        nh.param("useOpenCL", useOpenCL, false);
        getImageFunctionProvider()->initialize(useOpenCL, opencl_path, depthmask_cl);

        nh.param<std::string>("feature_detector", feature_detector, "ORB");
        nh.param<std::string>("feature_descriptor", feature_descriptor, "ORB");

        rmatcher->setFeatureDetector(feature_detector);
        rmatcher->setDescriptorExtractor(feature_descriptor);

        nh.param("tf_truth_initialize", tf_truth_initialize, false);
        nh.param<std::string>("tf_truth_topic", tf_truth_topic, "");
        nh.param<std::string>("calibration_pose", calib_frame_id_str, "");
        nh.param("tf_truth_init_time", tf_truth_init_time, -1);

        nh.param<std::string>("optical_parent", parent_frame_id_str, "optitrack");
        nh.param<std::string>("optical_frame", rgbd_frame_id_str, "rgbd_frame");
        nh.param<std::string>("depth_processing", depth_processing_str, "none");
        std::cout << "RGBD parent coordinate frame name = \"" << optical_parent << "\"" << std::endl;
        std::cout << "RGBD coordinate frame name =  \"" << optical_frame << "\"" << std::endl;

        if (depth_processing_str.compare("moving_average") == 0) {
            std::cout << "Applying moving average depth filter." << std::endl;
            this->depth_processing = Depth_Processing::MOVING_AVERAGE;
        } else if (depth_processing_str.compare("dither") == 0) {
            std::cout << "Applying dithering depth filter." << std::endl;
            this->depth_processing = Depth_Processing::DITHER;
        } else {
            this->depth_processing = Depth_Processing::NONE;
        }

        if (tf_truth_initialize) {
            std::cout << "Initializing transform to ground truth from topic \""
                    << tf_truth_topic << "\" wait " << tf_truth_init_time
                    << " seconds." << std::endl;
            sub_tf_truth = nh.subscribe(tf_truth_topic, 10, &RGBDOdometryEngine::tf_truth_Callback, this);
            if (tf_truth_init_time < 0) {
                tf_truth_init_time = 10;
            }
            std::cout << "Waiting for ground truth transform on topic \""
                    << tf_truth_topic << "\"..." << std::endl;
        } else {
            initializationDone = true;
            tf::Transform identity;
            identity.setIdentity();
            setInitialTransform(identity);
        }
    }

    virtual ~RGBDOdometryEngine() {
    }

    void initializeSubscribersAndPublishers();

    void setInitialTransform(tf::Transform target_pose, bool isOdomPose = false);

    void tf_truth_Callback(const geometry_msgs::TransformStampedConstPtr& tf_truth);
    // UNCOMMENT WHEN USING vrpn_client_ros 
    //    void tf_truth_Callback(const geometry_msgs::PoseStampedConstPtr& tf_truth);

    void rgbdImageCallback(const sensor_msgs::ImageConstPtr& depth_msg,
            const sensor_msgs::ImageConstPtr& rgb_msg_in,
            const sensor_msgs::CameraInfoConstPtr& info_msg);

    void tofRGBImageCallback(const sensor_msgs::ImageConstPtr& x_msg,
            const sensor_msgs::ImageConstPtr& y_msg,
            const sensor_msgs::ImageConstPtr& z_msg,
            const sensor_msgs::ImageConstPtr& rgb_msg,
            const sensor_msgs::ImageConstPtr& uv);

    void tofGreyImageCallback(const sensor_msgs::ImageConstPtr& depth_img,
            const sensor_msgs::ImageConstPtr& rgb_msg_in,
            const sensor_msgs::CameraInfoConstPtr& info_msg);

    void changePose(tf::Transform xform);
private:
    // -------------------------
    // Disabling default copy constructor and default
    // assignment operator.
    // -------------------------
    RGBDOdometryEngine(const RGBDOdometryEngine& yRef);
    RGBDOdometryEngine& operator=(const RGBDOdometryEngine& yRef);

    ros::NodeHandle nh;
    ros::NodeHandlePtr nodeptr;

    // variables held to process the current frame
    ros::Time frame_time;
    std::string frame_id_str;
    cv_bridge::CvImageConstPtr cv_rgbimg_ptr;
    cv_bridge::CvImageConstPtr cv_depthimg_ptr;
    image_geometry::PinholeCameraModel model_;
    // variables needed when using point cloud processing
    sensor_msgs::PointCloud2 frame_ptcloud;
    sensor_msgs::PointCloud2::Ptr ptcloud_sptr;
    //pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_ptcloud_sptr;
    std::string depth_encoding;
    //int depth_row_step;

    bool initializationDone;
    //cv::Ptr<std::vector<cv::KeyPoint> > prior_keypoints;
    //pcl::PointCloud<pcl::PointXYZRGB>::Ptr prior_ptcloud_sptr;
    std::string prior_keyframe_frameid_str;
    //bool fast_match;
    //int numKeyPoints;
    //RobustMatcher::Ptr rmatcher; // instantiate RobustMatcher

    // published odometry messages
    ros::Publisher pubXforms;
    ros::Publisher pubOdomMsg;
    ros::Publisher pubPose_w_cov;
    ros::Publisher pubOdom_w_cov;

    //    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::CameraInfo> MyApproximateSyncPolicy;
    //    message_filters::Synchronizer<MyApproximateSyncPolicy> syncApprox;

    // subscribers to RGBD sensor data
    image_transport::SubscriberFilter sub_depthImage;
    image_transport::SubscriberFilter sub_rgbImage;
    message_filters::Subscriber<sensor_msgs::CameraInfo> sub_rgbCameraInfo;

    // variables used when using a ground truth reference initialization
    int tf_truth_init_time;
    ros::Subscriber sub_tf_truth;

    std::string parent_frame_id_str; // frame id for parent of the RGBD sensor
    std::string rgbd_frame_id_str; // frame id for RGBD sensor      
    std::string calib_frame_id_str; // TF topic for calibration pose

    // variables to broadcast odometry to TF and track the RGBD sensor pose
    tf::TransformBroadcaster br;
    tf::Transform rgbd_pose;
};

#endif /* RGBD_ODOMETRY_H */

