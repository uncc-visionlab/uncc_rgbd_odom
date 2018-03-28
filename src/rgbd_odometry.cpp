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

// OpenCV includes
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// Eigen includes
#include <Eigen/Dense>
#include <Eigen/Geometry>

// PCL includes
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>


// Includes for this Library
#include <rgbd_odometry/rgbd_odometry.h>
#include <rgbd_odometry/point_cloud_xyzrgb.h>

// Draw only the 2D points
// For circles
//#define DEBUG false
//#define COMPUTE_PTCLOUDS false
//#define IMAGE_MASK_MARGIN 20
////#define PERFORMANCE_EVAL false
//
bool DUMP_MATCH_IMAGES = false;
bool DUMP_RAW_IMAGES = false;
bool SHOW_ORB_vs_iGRaND = false;

int toIndex(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, int column, int row) {
    return row * cloud->width + column;
}

void draw2DPoints(cv::UMat image, std::vector<cv::Point2f> &list_points, cv::Scalar color) {
    static int radius = 4;
    for (size_t i = 0; i < list_points.size(); i++) {
        cv::Point2f point_2d = list_points[i];
        // Draw Selected points
        cv::circle(image, point_2d, radius, color);
    }
}

void draw2DKeyPoints(cv::UMat image, std::vector<cv::KeyPoint> &keypoints, cv::Scalar color) {
    int radius;
    for (size_t i = 0; i < keypoints.size(); i++) {
        cv::Point2f point_2d = keypoints[i].pt;
        radius = 2 * keypoints[i].octave;
        // Draw Selected points
        cv::circle(image, point_2d, radius, color);
        cv::Point2f pt_dir(2 * radius * cos(keypoints[i].angle * M_PI / 180), 2 * radius * sin(keypoints[i].angle * M_PI / 180));
        pt_dir.x += point_2d.x;
        pt_dir.y += point_2d.y;
        cv::line(image, point_2d, pt_dir, color);
    }
}

pcl::PointXYZRGB convertRGBD2XYZ(cv::Point2f point2d_frame, cv::Mat rgb_img, cv::Mat depth_img,
        image_geometry::PinholeCameraModel model_, int row_step) {

    float u = point2d_frame.x;
    float v = point2d_frame.y;
    int iu = round(u);
    int iv = round(v);
    int offset = (iv * depth_img.cols + iu);
    //int offset = (iv * (row_step/sizeof(float)) + iu);
    float depth = ((float *) depth_img.data)[offset];
    float center_x = model_.cx();
    float center_y = model_.cy();
    // Combine unit conversion (if necessary) with scaling by focal length for computing (X,Y)
    double unit_scaling = 1;
    float constant_x = unit_scaling / model_.fx();
    float constant_y = unit_scaling / model_.fy();
    float bad_point = std::numeric_limits<float>::quiet_NaN();
    float x, y, z;
    if (!std::isfinite(depth)) {
        x = y = z = bad_point;
        std::cout << "CONVERTING BAD (NAN) KEYPOINT at location (" << iu << ", " << iv << ")!" << std::endl;
    } else {
        // Fill in XYZ
        x = (iu - center_x) * depth * constant_x;
        y = (iv - center_y) * depth * constant_y;
        z = depth;
    }
    uchar r = rgb_img.data[rgb_img.channels()*(iv * rgb_img.cols + iu) + 0];
    uchar g = rgb_img.data[rgb_img.channels()*(iv * rgb_img.cols + iu) + 1];
    uchar b = rgb_img.data[rgb_img.channels()*(iv * rgb_img.cols + iu) + 2];
    pcl::PointXYZRGB pt(r, g, b);
    pt.x = x;
    pt.y = y;
    pt.z = z;
    return pt;
}

bool RGBDOdometryEngine::computeRelativePose2(std::string& name,
        cv::Ptr<cv::FeatureDetector> detector_,
        cv::Ptr<cv::DescriptorExtractor> extractor_, Eigen::Matrix4f& trans,
        Eigen::Map<Eigen::Matrix<double, 6, 6> >& covMatrix,
        cv::UMat& frame,
        cv::Ptr<std::vector<cv::KeyPoint> >& keypoints_frame,
        cv::Ptr<cv::UMat>& descriptors_frame,
        cv::UMat& prior_frame,
        cv::Ptr<std::vector<cv::KeyPoint> >& prior_keypoints_frame,
        cv::Ptr<cv::UMat>& prior_descriptors_frame,
        std::vector<Eigen::Matrix4f>& transform_vector,
        float& detector_time, float& descriptor_time, float& match_time,
        float& RANSAC_time, float& covarianceTime,
        int& numFeatures, int& numMatches, int& numInliers) {

    return true;
}

bool RGBDOdometryEngine::computeRelativePose(std::string& name,
        cv::Ptr<cv::FeatureDetector> detector_,
        cv::Ptr<cv::DescriptorExtractor> extractor_, Eigen::Matrix4f& trans,
        Eigen::Map<Eigen::Matrix<double, 6, 6> >& covMatrix, cv::UMat& frame,
        cv::Ptr<std::vector<cv::KeyPoint> >& keypoints_frame,
        cv::Ptr<cv::UMat>& descriptors_frame,
        std::vector<Eigen::Matrix4f>& transform_vector,
        float& detector_time, float& descriptor_time, float& match_time,
        float& RANSAC_time, float& covarianceTime,
        int& numFeatures, int& numMatches, int& numInliers) {
    static int bad_frames = 0;
    //    ROS_DEBUG("An image was received.\n");
    // unload data from the messenger class
    std::string keyframe_frameid_str = this->frame_id_str;
    //        sensor_msgs::PointCloud2::Ptr ptcloud_sptr = aptr->getROSPointCloud2();
    //    if (!COMPUTE_PTCLOUDS) {
    //        pcl_ptcloud_sptr.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
    //        pcl_ptcloud_sptr->is_dense = true;
    //    }
#ifdef OPENCV3
    cv::UMat depth_frame;
    cv::UMat frame_vis = frame.clone(); // refresh visualization frame
    cv::UMat mask;
#else
    cv::Mat depth_frame = cv_depthimg_ptr->image.getUMat(cv::ACCESS_READ);
    cv::Mat frame_vis = frame.clone(); // refresh visualization frame
    cv::Mat mask; // type of mask is CV_8U
#endif
    // Convert Kinect depth image from image-of-shorts (mm) to image-of-floats (m)
    if (depth_encoding == sensor_msgs::image_encodings::TYPE_16UC1) {
        ROS_DEBUG("Converting Kinect-style depth image to floating point depth image.");
        int width = cv_depthimg_ptr->image.cols;
        int height = cv_depthimg_ptr->image.rows;
        depth_frame.create(height, width, CV_32F);
        float bad_point = std::numeric_limits<float>::quiet_NaN();
        uint16_t* uint_depthvals = (uint16_t *) cv_depthimg_ptr->image.data;
        float* float_depthvals = (float *) depth_frame.getMat(cv::ACCESS_WRITE).data;
        for (int row = 0; row < height; ++row) {
            for (int col = 0; col < width; ++col) {
                if (uint_depthvals[row * width + col] == 0) {
                    float_depthvals[row * width + col] = bad_point;
                } else { // mm to m for kinect
                    float_depthvals[row * width + col] = uint_depthvals[row * width + col]*0.001f;
                }
            }
        }
    } else if (depth_encoding == sensor_msgs::image_encodings::TYPE_32FC1) {
        depth_frame = cv_depthimg_ptr->image.getUMat(cv::ACCESS_READ);
    }

    getImageFunctionProvider()->computeMask(depth_frame, mask);
    cv::Mat filtered_depth;
    cv::UMat smoothed_depth_frame;
    switch (depth_processing) {
        case Depth_Processing::MOVING_AVERAGE:
            imageFunctionProvider->movingAvgFilter(depth_frame, smoothed_depth_frame,
                    keyframe_frameid_str);
            smoothed_depth_frame.copyTo(depth_frame);
            break;
        case Depth_Processing::DITHER:
            imageFunctionProvider->ditherDepthAndSmooth(depth_frame, smoothed_depth_frame,
                    keyframe_frameid_str);
            smoothed_depth_frame.copyTo(depth_frame);
            break;
        case Depth_Processing::NONE:
        default:
            break;
    }

    cv::Mat dimg = depth_frame.getMat(cv::ACCESS_READ);
    if (cv::iGRAND * iGRAND_detector = rmatcher->detector_.dynamicCast<cv::iGRAND>()) {
        iGRAND_detector->setDepthImage(&dimg);
    }
    double t = (double) cv::getTickCount();
    detector_->detect(frame, *keypoints_frame, mask);
    numFeatures = keypoints_frame->size();
#ifdef PERFORMANCE_EVAL
    if (keypoints_frame->size() > MAX_KEYPOINTS) {
        keypoints_frame->resize(MAX_KEYPOINTS);
    }
#endif    
    detector_time = (cv::getTickCount() - t) * 1000. / cv::getTickFrequency();
    //printf("execution time = %dms\n", (int) (t * 1000. / cv::getTickFrequency()));
    if (SHOW_ORB_vs_iGRaND) {
        if (name.compare("ORB") == 0) {
            cv::Scalar red(0, 0, 255); // BGR order
            draw2DKeyPoints(frame_vis, *keypoints_frame, red);
            //cv::imwrite(keyframe_frameid_str + "_" + name + "_keypoints.png", frame_vis);
            cv::imshow("ORB Feature Detections (red)", frame_vis); // Show our image inside it.
            cv::waitKey(3);
        } else if (name.compare("iGRAND") == 0) {
            cv::Scalar green(0, 255, 0); // BGR order            
            draw2DKeyPoints(frame_vis, *keypoints_frame, green);
            cv::imwrite(keyframe_frameid_str + "_" + name + "_keypoints.png", frame_vis);
            cv::imshow("iGRaND Feature Detections (green)", frame_vis); // Show our image inside it.
            cv::waitKey(3);
        }
        //cv::imshow("Feature Detections (red)", frame_vis); // Show our image inside it.
        //cv::waitKey(3);
    }

    if (!COMPUTE_PTCLOUDS) {
        int i = 0;
        std::cout << "Found " << keypoints_frame->size() << " key points in frame." << std::endl;
        std::vector<cv::KeyPoint>::iterator keyptIterator;
        for (keyptIterator = keypoints_frame->begin();
                keyptIterator != keypoints_frame->end(); /*++keyptIterator*/) {
            cv::KeyPoint kpt = *keyptIterator;
            int offset = (round(kpt.pt.y) * depth_frame.cols + round(kpt.pt.x));
            if (((int) mask.getMat(cv::ACCESS_READ).data[offset]) == 0) {
                //float depth = ((float *) depth_frame.getMat(cv::ACCESS_READ).data)[offset];
                //                std::cout << "mask = " << ((int) mask.getMat(cv::ACCESS_READ).data[offset])
                //                        << " depth = " << depth << std::endl;
                // key point found that lies in/too close to the masked region!!
                keyptIterator = keypoints_frame->erase(keyptIterator);
                continue;
            } else {
                ++keyptIterator;
            }
            pcl::PointXYZRGB pt;
            pt = convertRGBD2XYZ(kpt.pt, frame.getMat(cv::ACCESS_READ),
                    dimg, model_, depth_row_step);
            //std::cout << "Added point (" << pt.x << ", " << pt.y << ", " << pt.z << ")" << std::endl;
            pcl_ptcloud_sptr->push_back(pt);
            if (std::isnan(kpt.pt.x) || std::isnan(kpt.pt.y) ||
                    std::isnan(pt.x) || std::isnan(pt.y) || std::isnan(pt.z)) {
                ROS_INFO("%d : 2d (x,y)=(%f,%f)  mask(x,y)=%d (x,y,z)=(%f,%f,%f)\n",
                        i++, kpt.pt.x, kpt.pt.y,
                        mask.getMat(cv::ACCESS_READ).data[offset],
                        pt.x, pt.y, pt.z);
            }
        }
    }
    if (keypoints_frame->size() < 10) {
        ROS_DEBUG("Too few keypoints! Bailing on image...");
        bad_frames++;
        if (bad_frames > 2) {
            ROS_DEBUG(" and Re-initializing the estimator.");
            prior_descriptors_.release();
        }
        return false;
    }
    if (cv::iGRAND * iGRAND_extractor = extractor_.dynamicCast<cv::iGRAND>()) {
        iGRAND_extractor->setDepthImage(&dimg);
    }
    t = (double) cv::getTickCount();
    extractor_->compute(frame, *keypoints_frame, *descriptors_frame);
    descriptor_time = (cv::getTickCount() - t) * 1000. / cv::getTickFrequency();
    match_time = 0;
    t = (double) cv::getTickCount();
    //    if (!prior_descriptors_ || prior_descriptors_.size() < 6) {
    if (!prior_descriptors_ || prior_descriptors_->empty()) {
        return false;
    }
    // list with descriptors of each 3D coordinate
    t = (double) cv::getTickCount();
    std::vector<cv::DMatch> good_matches; // to obtain the 3D points of the model

    if (fast_match) {
        rmatcher->fastRobustMatch(good_matches, *prior_descriptors_, *descriptors_frame);
    } else {
        rmatcher->robustMatch(good_matches, *prior_descriptors_, *descriptors_frame);
    }
    std::cout << "from (" << prior_keypoints->size() << "," << keypoints_frame->size() << ")"
            << " key points found " << good_matches.size() << " good matches." << std::endl;
    match_time = (cv::getTickCount() - t) * 1000. / cv::getTickFrequency();
    numMatches = good_matches.size();
    if (good_matches.size() < 6) {
        ROS_DEBUG("Too few key point matches in the images! Bailing on image...");
        bad_frames++;
        if (bad_frames > 2) {
            ROS_DEBUG(" and Re-initializing the estimator.");
            prior_descriptors_.release();
        }
        return false;
    }
    if (DUMP_MATCH_IMAGES && !prior_image.empty()) {
        cv::UMat matchImage;
        cv::Mat pImage = prior_image.getMat(cv::ACCESS_WRITE);
        //std::cout << "priorImg(x,y)=(" << pImage.cols << ", " << pImage.rows << ")" << std::endl;
        // select a region of interest
        cv::Mat pRoi = pImage(cv::Rect(620, 0, 20, 480));

        // set roi to some rgb colour   
        pRoi.setTo(cv::Scalar(255, 255, 255));

        pImage = frame.getMat(cv::ACCESS_WRITE);
        // select a region of interest
        //std::cout << "priorImg2(x,y)=(" << pImage2.cols << ", " << pImage2.rows << ")" << std::endl;
        pRoi = pImage(cv::Rect(0, 0, 20, 480));

        // set roi to some rgb colour   
        pRoi.setTo(cv::Scalar(255, 255, 255));

        cv::drawMatches(prior_image, *prior_keypoints, frame, *keypoints_frame, good_matches,
                matchImage, cv::Scalar::all(-1), cv::Scalar::all(-1),
                std::vector<char>(), cv::DrawMatchesFlags::DEFAULT);
        //                    std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        try {
            cv::imshow("Display window", matchImage); // Show our image inside it.
            cv::waitKey(3);
            //cv::imwrite(keyframe_frameid_str + "_matches.png", matchImage);
        } catch (cv_bridge::Exception& e) {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return false;
        }
    }
    // -- Step 2: Find out the 2D/3D correspondences
    std::vector<cv::Point2f> list_points2d_prior_match; // container for the model 2D coordinates found in the scene
    std::vector<cv::Point2f> list_points2d_scene_match; // container for the model 2D coordinates found in the scene
    //            std::vector<int> list_indices_prior_match;
    //            std::vector<int> list_indices_scene_match;
    //            pcl::PointCloud<pcl::PointXYZ>::Ptr prior_pts_xyz(new pcl::PointCloud<pcl::PointXYZ>());
    //            pcl::PointCloud<pcl::PointXYZ>::Ptr frame_pts_xyz(new pcl::PointCloud<pcl::PointXYZ>());        
    pcl::CorrespondencesPtr ptcloud_matches(new pcl::Correspondences());
    // each matched feature is placed into a point cloud
    for (unsigned int match_index = 0; match_index < good_matches.size(); ++match_index) {
        cv::Point2f point2d_prior = (*prior_keypoints)[ good_matches[match_index].queryIdx ].pt; // 2D point from model
        cv::Point2f point2d_frame = (*keypoints_frame)[ good_matches[match_index].trainIdx ].pt; // 2D point from the scene
        //                list_points2d_prior_match.push_back(point2d_prior); // add 3D point
        //                list_points2d_scene_match.push_back(point2d_frame); // add 2D point
        if (false && (std::isnan(point2d_frame.x) || std::isnan(point2d_frame.y) ||
                point2d_frame.x > pcl_ptcloud_sptr->width ||
                point2d_frame.y > pcl_ptcloud_sptr->height)) {

            ROS_INFO("Frame coord out of range for index pair (%d,%d) ! (x,y)=(%f,%f) (width,height)=(%d,%d)",
                    good_matches[match_index].trainIdx, good_matches[match_index].queryIdx,
                    point2d_frame.x, point2d_frame.y,
                    pcl_ptcloud_sptr->width, pcl_ptcloud_sptr->height);
        }
        if (COMPUTE_PTCLOUDS) {
            pcl::Correspondence correspondence(toIndex(pcl_ptcloud_sptr, point2d_frame.x, point2d_frame.y),
                    toIndex(prior_ptcloud_sptr, point2d_prior.x, point2d_prior.y),
                    good_matches[match_index].distance);
            ptcloud_matches->push_back(correspondence);
        } else {
            // std::cout << "distance " << good_matches[match_index].distance << std::endl;
            pcl::Correspondence correspondence(good_matches[match_index].trainIdx,
                    good_matches[match_index].queryIdx,
                    good_matches[match_index].distance);
            ptcloud_matches->push_back(correspondence);
        }
    }

#if (DEBUG==true)
    Eigen::Vector4f xyz_centroid;
    Eigen::Matrix3f covariance_matrix = Eigen::Matrix3f::Zero();
    //        pcl::PointCloud<pcl::PointXYZRGB>::iterator ptIter;
    //        for (ptIter = pcl_ptcloud_sptr->points.begin(); ptIter != pcl_ptcloud_sptr->points.end();
    //                ++ptIter) {
    //        for (int ptIdx = 0; ptIdx < pcl_ptcloud_sptr->size(); ptIdx++) {
    //            pcl::PointXYZRGB pt = *ptIter;
    //            pcl::PointXYZRGB pt = (*pcl_ptcloud_sptr)[ptIdx];
    //            std::cout << ptIdx << " : " << pt.x << "," << pt.y << "," << pt.z << ", " << pt.rgba << std::endl;
    //        }

    pcl::computeMeanAndCovarianceMatrix(*pcl_ptcloud_sptr, covariance_matrix, xyz_centroid);
    std::cout << "prior size = " << prior_ptcloud_sptr->size() << " frame size = " << pcl_ptcloud_sptr->size() << std::endl;
    Eigen::toString("mean", xyz_centroid);
    Eigen::toString("cov", covariance_matrix);
#endif

    t = (double) cv::getTickCount();
    pcl::registration::CorrespondenceRejectorSampleConsensus<pcl::PointXYZRGB>::Ptr ransac_rejector(
            new pcl::registration::CorrespondenceRejectorSampleConsensus<pcl::PointXYZRGB>);
    ransac_rejector->setInputSource(pcl_ptcloud_sptr);
    ransac_rejector->setInputTarget(prior_ptcloud_sptr);
    ransac_rejector->setInlierThreshold(0.05);
    ransac_rejector->setRefineModel(true);
    ransac_rejector->setInputCorrespondences(ptcloud_matches);
    ransac_rejector->setMaximumIterations(400);
    pcl::CorrespondencesPtr ptcloud_matches_ransac(new pcl::Correspondences());
    ransac_rejector->getRemainingCorrespondences(*ptcloud_matches, *ptcloud_matches_ransac);
    if (ptcloud_matches_ransac->size() < 2) {
        ROS_DEBUG("Too few inliers from RANSAC transform estimation! Bailing on image...");
        bad_frames++;
        if (bad_frames > 2) {
            ROS_DEBUG(" and Re-initializing the estimator.");
            prior_descriptors_.release();
        }
        return false;
    }
    trans = ransac_rejector->getBestTransformation();
    RANSAC_time = (cv::getTickCount() - t) * 1000. / cv::getTickFrequency();
    std::cout << "RANSAC rejection left " << ptcloud_matches_ransac->size() << " inliers." << std::endl;
    numInliers = ptcloud_matches_ransac->size();
    //Eigen::toString("trans = ", trans);
    pcl::registration::TransformationEstimationSVD<pcl::PointXYZRGB, pcl::PointXYZRGB> trans_est;
    //        Eigen::Matrix4f transformVal;
    //        trans_est.estimateRigidTransformation(*pcl_ptcloud_sptr, *prior_ptcloud_sptr,
    //                *ptcloud_matches_ransac, transformVal);
    //        Eigen::toString("transform = ", transformVal);
    pcl::Correspondences::iterator correspondenceIterator;
    pcl::CorrespondencesPtr src2tgt_correspondences(new pcl::Correspondences());
    int idxval = 0;
    for (correspondenceIterator = ptcloud_matches_ransac->begin();
            correspondenceIterator != ptcloud_matches_ransac->end();
            ++correspondenceIterator) {
        pcl::Correspondence match3Didxs = *correspondenceIterator;
        src2tgt_correspondences->push_back(pcl::Correspondence(idxval, idxval, match3Didxs.distance));
        idxval++;
    }

    //        cv::Point2f point2d_prior = (*prior_keypoints)[ good_matches[match_index].queryIdx ].pt; // 2D point from model
    //        cv::Point2f point2d_frame = (*keypoints_frame)[ good_matches[match_index].trainIdx ].pt; // 2D point from the scene
    //
    //        pcl::Correspondence correspondence(good_matches[match_index].trainIdx,
    //                good_matches[match_index].queryIdx,
    //                good_matches[match_index].distance);
    //        std::vector<cv::Point2f> ransac_pixels_prior;
    //        std::vector<cv::Point2f> ransac_pixels_frame;
    // has indices of corresponding 3D points out of RANSAC
    //        for (correspondenceIterator = ptcloud_matches_ransac->begin();
    //                correspondenceIterator != ptcloud_matches_ransac->end();
    //                ++correspondenceIterator) {
    //            pcl::Correspondence match3Didxs = *correspondenceIterator;
    //            cv::Point2f pixel_prior = (*prior_keypoints)[match3Didxs.index_match];
    //            cv::Point2f pixel_frame = (*keypoints_frame)[match3Didxs.index_query];
    //            ransac_pixels_prior.push_back(pixel_prior);
    //            ransac_pixels_frame.push_back(pixel_frame);
    //        }
    t = (double) cv::getTickCount();
    int MAX_TRIALS = 200;
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.0f, 1.0f);
    pcl::PointCloud<pcl::PointXYZRGB> srcCloud;
    pcl::PointCloud<pcl::PointXYZRGB> tgtCloud;
    //std::vector<Eigen::Matrix4f> transform_vector;
    //        for (correspondenceIterator = ptcloud_matches_ransac->begin();
    //                correspondenceIterator != ptcloud_matches_ransac->end();
    //                ++correspondenceIterator) {
    //            pcl::Correspondence match3Didxs = *correspondenceIterator;
    //            pcl::PointXYZRGB pt3_rgb_target = prior_ptcloud_sptr->points[match3Didxs.index_match];
    //            pcl::PointXYZRGB pt3_rgb_source = pcl_ptcloud_sptr->points[match3Didxs.index_query];
    //            cv::Point2f& pixel_prior = (*prior_keypoints)[match3Didxs.index_match].pt;
    //            cv::Point2f& pixel_frame = (*keypoints_frame)[match3Didxs.index_query].pt;
    //            std::cout << " pt3_prev = " << pt3_rgb_target.x << ", " << pt3_rgb_target.y << ", " << pt3_rgb_target.z << std::endl;
    //            std::cout << " pt3_cur = " << pt3_rgb_source.x << ", " << pt3_rgb_source.y << ", " << pt3_rgb_source.z << std::endl;
    //            std::cout << " pt2_prev = " << pixel_prior.x << ", " << pixel_prior.y << std::endl;
    //            std::cout << " pt2_cur = " << pixel_frame.x << ", " << pixel_frame.y << std::endl;
    //        }
    Eigen::MatrixXd estimate_matrix(MAX_TRIALS, 6);
    for (int trials = 0; trials < MAX_TRIALS; trials++) {
        for (correspondenceIterator = ptcloud_matches_ransac->begin();
                correspondenceIterator != ptcloud_matches_ransac->end();
                ++correspondenceIterator) {
            pcl::Correspondence match3Didxs = *correspondenceIterator;
            pcl::PointXYZRGB pt3_rgb_target = prior_ptcloud_sptr->points[match3Didxs.index_match];
            pcl::PointXYZRGB pt3_rgb_source = pcl_ptcloud_sptr->points[match3Didxs.index_query];
            cv::Point2f& pixel_prior = (*prior_keypoints)[match3Didxs.index_match].pt;
            cv::Point2f& pixel_frame = (*keypoints_frame)[match3Didxs.index_query].pt;

            float depth_sq = pt3_rgb_target.z * pt3_rgb_target.z;
            float std_normal_sample = distribution(generator) * (1.425e-3f) * depth_sq;
            pt3_rgb_target.x += std_normal_sample * ((pixel_prior.x - model_.cx()) / model_.fx());
            pt3_rgb_target.y += std_normal_sample * ((pixel_prior.y - model_.cy()) / model_.fy());
            pt3_rgb_target.z += std_normal_sample;

            depth_sq = pt3_rgb_source.z * pt3_rgb_source.z;
            std_normal_sample = distribution(generator) * (1.425e-3f) * depth_sq;
            pt3_rgb_source.x += std_normal_sample * ((pixel_frame.x - model_.cx()) / model_.fx());
            pt3_rgb_source.y += std_normal_sample * ((pixel_frame.y - model_.cy()) / model_.fy());
            pt3_rgb_source.z += std_normal_sample;

            tgtCloud.push_back(pt3_rgb_target);
            srcCloud.push_back(pt3_rgb_source);
        }
        Eigen::Matrix4f noisy_transform;
        trans_est.estimateRigidTransformation(srcCloud, tgtCloud,
                *src2tgt_correspondences, noisy_transform);
        estimate_matrix(trials, 0) = noisy_transform(0, 3);
        estimate_matrix(trials, 1) = noisy_transform(1, 3);
        estimate_matrix(trials, 2) = noisy_transform(2, 3);
        estimate_matrix(trials, 3) = noisy_transform(0, 2);
        estimate_matrix(trials, 4) = noisy_transform(1, 2);
        estimate_matrix(trials, 5) = noisy_transform(0, 1);
        transform_vector.push_back(noisy_transform);
        //Eigen::toString("noisy_transform = ", noisy_transform);
        srcCloud.clear();
        tgtCloud.clear();
    }
    //double cov[36];
    //Eigen::Map<Eigen::Matrix<double, 6, 6> > covMatrix(cov);
    Eigen::MatrixXd centered = estimate_matrix.rowwise() - estimate_matrix.colwise().mean();
    covMatrix = (centered.adjoint() * centered) / double(estimate_matrix.rows() - 1);
    covarianceTime = (cv::getTickCount() - t) * 1000. / cv::getTickFrequency();

    if (false) {
        //                pcl::registration::TransformationEstimationSVD<pcl::PointXYZRGB, pcl::PointXYZRGB> TESVD;
        //                pcl::registration::TransformationEstimationSVD<pcl::PointXYZRGB, pcl::PointXYZRGB>::Matrix4 transformation2;
        //                TESVD.estimateRigidTransformation(*pcl_ptcloud_sptr, list_indices_scene_match,
        //                        *prior_ptcloud, list_indices_prior_match, transformation2);
        //                std::cout << transformation2(0, 3) << "," <<
        //                        transformation2(1, 3) << "," <<
        //                        transformation2(2, 3);

        pcl::SampleConsensusModelRegistration<pcl::PointXYZRGB>::Ptr ransac_model(
                new pcl::SampleConsensusModelRegistration<pcl::PointXYZRGB>(pcl_ptcloud_sptr));
        pcl::RandomSampleConsensus<pcl::PointXYZRGB> ransac(ransac_model);
        ransac.setDistanceThreshold(0.1);
        ransac.setMaxIterations(5);
        //                ransac_model->setIndices(list_indices_scene_match);
        //                ransac_model->setInputTarget(prior_ptcloud, list_indices_prior_match);
        //upping the verbosity level to see some info
        pcl::console::VERBOSITY_LEVEL vblvl = pcl::console::getVerbosityLevel();
        pcl::console::setVerbosityLevel(pcl::console::L_DEBUG);
        ransac.computeModel(1);
        pcl::console::setVerbosityLevel(vblvl);
        Eigen::VectorXf model_coefficients;
        ransac.getModelCoefficients(model_coefficients);
        //Eigen::toString("model_coeffs", model_coefficients);
    }
    if (DUMP_RAW_IMAGES) {
        cv::Scalar blue(255, 0, 0);
        cv::Scalar red(128, 0, 0);
        if (list_points2d_prior_match.size() > 0) {
            //                draw2DPoints(frame_vis, list_points2d_model_match, blue);
            //                draw2DPoints(depth_frame, list_points2d_model_match, blue);
            //                draw2DPoints(mask, list_points2d_model_match, blue);
            draw2DPoints(frame_vis, list_points2d_scene_match, red);
            draw2DPoints(depth_frame, list_points2d_scene_match, red);
            draw2DPoints(mask, list_points2d_scene_match, red);
        }
        try {
            cv::imwrite(keyframe_frameid_str + ".png", frame_vis);
            cv::imwrite(keyframe_frameid_str + "_depth.png", depth_frame);
            cv::imwrite(keyframe_frameid_str + "_mask.png", mask);
        } catch (cv_bridge::Exception& e) {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return true;
        }
    }
}

