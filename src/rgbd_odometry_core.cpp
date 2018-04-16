/* 
 * File:   rgbd_odometry_core.cpp
 * Author: arwillis
 *
 * Created on April 14, 2018, 2:34 PM
 */
#include <iostream>
#include <fstream>
#include <string>
#include <iterator>
#include <random>

#include <rgbd_odometry/rgbd_odometry_core.h>

bool DUMP_MATCH_IMAGES = false;
bool DUMP_RAW_IMAGES = false;
bool SHOW_ORB_vs_iGRaND = false;

#define IMAGE_MASK_MARGIN 20
//#define PERFORMANCE_EVAL false

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
        cv::Mat& rgbCamera_Kmatrix) {

    float u = point2d_frame.x;
    float v = point2d_frame.y;
    int iu = round(u);
    int iv = round(v);
    int offset = (iv * depth_img.cols + iu);
    float depth = ((float *) depth_img.data)[offset];
    float center_x = rgbCamera_Kmatrix.at<float>(0, 2);
    float center_y = rgbCamera_Kmatrix.at<float>(1, 2);
    // Combine unit conversion (if necessary) with scaling by focal length for computing (X,Y)
    double unit_scaling = 1;
    float constant_x = unit_scaling / rgbCamera_Kmatrix.at<float>(0, 0);
    float constant_y = unit_scaling / rgbCamera_Kmatrix.at<float>(1, 1);
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

void logTransformData(//std::string& frameid, ros::Time& frame_time,
        std::string& detector, std::string& descriptor,
        float detectorTime, float descriptorTime, float matchTime, float RANSACTime, float covarianceTime,
        int numFeatures, int numMatches, int numInliers,
        Eigen::Quaternionf& quat, Eigen::Vector3f& trans,
        Eigen::Matrix4f& transform, Eigen::Map<Eigen::Matrix<double, 6, 6> > covMatrix,
        std::vector<Eigen::Matrix4f>& transform_vector) {
    static int elementIdx = 1;
    static int StreamPrecision = 8;
    static Eigen::IOFormat OctaveFmt(StreamPrecision, 0, ", ", ";\n", "", "", "[", "]");

    //if (transform_vector.size() < 1 || frameid.size() < 1)
    //    return;
    try {
        if (!fos) {
            std::cout << "Opening logfile " << _logfilename << "." << std::endl;
            fos.open(_logfilename.c_str());
        }
        Eigen::Matrix4f curr_transform;
        std::vector<Eigen::Matrix4f>::iterator poseIterator;
        //fos << "rgbd_odometry{" << elementIdx << "}.frame_id = '" << frameid << "';" << std::endl;
        //fos << "rgbd_odometry{" << elementIdx << "}.sec = " << frame_time.sec << ";" << std::endl;
        //fos << "rgbd_odometry{" << elementIdx << "}.nsec = " << frame_time.nsec << ";" << std::endl;
        fos << "rgbd_odometry{" << elementIdx << "}.detector = '" << detector << "';" << std::endl;
        fos << "rgbd_odometry{" << elementIdx << "}.descriptor = '" << descriptor << "';" << std::endl;
        fos << "rgbd_odometry{" << elementIdx << "}.detectorTime = " << detectorTime << ";" << std::endl;
        fos << "rgbd_odometry{" << elementIdx << "}.descriptorTime = " << descriptorTime << ";" << std::endl;
        fos << "rgbd_odometry{" << elementIdx << "}.matchTime = " << matchTime << ";" << std::endl;
        fos << "rgbd_odometry{" << elementIdx << "}.RANSACTime = " << RANSACTime << ";" << std::endl;
        fos << "rgbd_odometry{" << elementIdx << "}.covarianceTime = " << covarianceTime << ";" << std::endl;
        fos << "rgbd_odometry{" << elementIdx << "}.numFeatures = " << numFeatures << ";" << std::endl;
        fos << "rgbd_odometry{" << elementIdx << "}.numMatches = " << numMatches << ";" << std::endl;
        fos << "rgbd_odometry{" << elementIdx << "}.numInliers = " << numInliers << ";" << std::endl;
        fos << "rgbd_odometry{" << elementIdx << "}.quaternion = [";
        fos << quat.x() << " " << quat.y() << " " << quat.z() << " " << quat.w() << "];" << std::endl;
        fos << "rgbd_odometry{" << elementIdx << "}.translation = [";
        fos << trans.x() << " " << trans.y() << " " << trans.z() << "];" << std::endl;
        fos << "rgbd_odometry{" << elementIdx << "}.transform = ";
        fos << transform.format(OctaveFmt) << ";" << std::endl;
        fos << "rgbd_odometry{" << elementIdx << "}.covMatrix = ";
        fos << covMatrix.format(OctaveFmt) << ";" << std::endl;
        //        fos << "rgbd_odometry{" << elementIdx << "}.noisy_transforms = [ " << std::endl;
        //        for (poseIterator = transform_vector.begin();
        //                poseIterator != transform_vector.end(); ++poseIterator) {
        //            curr_transform = *poseIterator;
        //            fos << curr_transform.format(OctaveFmt) << ";" << std::endl;
        //        }
        //        fos << "];" << std::endl;
        elementIdx++;
    } catch (...) {
        printf("error writing to noisy transform log file.");
        return;
    }
}

bool RGBDOdometryCore::compute(cv::UMat &frame, cv::UMat &depthimg) {

    bool odomEstimatorSuccess;
    float detectorTime, descriptorTime, matchTime, RANSACTime, covarianceTime;
    std::vector<Eigen::Matrix4f> transform_vector;
    cv::Ptr<std::vector<cv::KeyPoint> > keypoints_frame(new std::vector<cv::KeyPoint>);
    cv::Ptr<cv::UMat> descriptors_frame(new cv::UMat);
    double cov[36];
    Eigen::Map<Eigen::Matrix<double, 6, 6> > covMatrix(cov);
    Eigen::Matrix4f trans;
    int numFeatures = 0, numMatches = 0, numInliers = 0;
    transform_vector.clear();

    //std::cout << "Detector = " << detectorStr << " Descriptor = " << descriptorStr << std::endl;
    odomEstimatorSuccess = computeRelativePose(rmatcher->detectorStr,
            rmatcher->detector_, rmatcher->extractor_, trans, covMatrix,
            depthimg, frame, keypoints_frame, descriptors_frame,
            transform_vector,
            detectorTime, descriptorTime, matchTime, RANSACTime, covarianceTime,
            numFeatures, numMatches, numInliers);

    prior_keypoints = keypoints_frame;
    prior_descriptors_ = descriptors_frame;
    prior_ptcloud_sptr = pcl_ptcloud_sptr;
    if (!odomEstimatorSuccess) {
        return false;
    }
    Eigen::Quaternionf quat(trans.block<3, 3>(0, 0));
    Eigen::Vector3f translation(trans.block<3, 1>(0, 3));
    if (LOG_ODOMETRY_TO_FILE) {
        logTransformData(//keyframe_frameid_str, frame_time,
                rmatcher->detectorStr, rmatcher->descriptorStr,
                detectorTime, descriptorTime, matchTime, RANSACTime, covarianceTime,
                numFeatures, numMatches, numInliers,
                quat, translation,
                trans, covMatrix, transform_vector);
    }
    //prior_keyframe_frameid_str = keyframe_frameid_str;
    prior_image = frame.clone();
}

int RGBDOdometryCore::computeKeypointsAndDescriptors(cv::UMat& frame, cv::Mat& dimg, cv::UMat& mask,
        std::string& name,
        cv::Ptr<cv::FeatureDetector> detector_,
        cv::Ptr<cv::DescriptorExtractor> extractor_,
        cv::Ptr<std::vector<cv::KeyPoint> >& keypoints_frame,
        cv::Ptr<cv::UMat>& descriptors_frame, float& detector_time, float& descriptor_time,
        const std::string keyframe_frameid_str) {
#ifdef USE_iGRAND
    if (cv::iGRAND * iGRAND_detector = rmatcher->detector_.dynamicCast<cv::iGRAND>()) {
        iGRAND_detector->setDepthImage(&dimg);
    }
#endif

    double t = (double) cv::getTickCount();
    detector_->detect(frame, *keypoints_frame, mask);
    int numFeatures = keypoints_frame->size();
#ifdef PERFORMANCE_EVAL
    if (keypoints_frame->size() > MAX_KEYPOINTS) {
        keypoints_frame->resize(MAX_KEYPOINTS);
    }
#endif    
    detector_time = (cv::getTickCount() - t) * 1000. / cv::getTickFrequency();
    //printf("execution time = %dms\n", (int) (t * 1000. / cv::getTickFrequency()));
    if (SHOW_ORB_vs_iGRaND) {
        cv::UMat frame_vis = frame.clone(); // refresh visualization frame
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

#ifdef USE_iGRAND
    if (cv::iGRAND * iGRAND_extractor = extractor_.dynamicCast<cv::iGRAND>()) {
        iGRAND_extractor->setDepthImage(&dimg);
    }
#endif

    t = (double) cv::getTickCount();
    extractor_->compute(frame, *keypoints_frame, *descriptors_frame);
    descriptor_time = (cv::getTickCount() - t) * 1000. / cv::getTickFrequency();
    return numFeatures;
}

bool RGBDOdometryCore::estimateCovarianceBootstrap(pcl::CorrespondencesPtr ptcloud_matches_ransac,
        cv::Ptr<std::vector<cv::KeyPoint> >& keypoints_frame,
        cv::Ptr<std::vector<cv::KeyPoint> >& prior_keypoints,
        Eigen::Map<Eigen::Matrix<double, 6, 6> >& covMatrix,
        std::vector<Eigen::Matrix4f>& transform_vector,
        float &covarianceTime) {
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
    double t = (double) cv::getTickCount();
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
    float center_x = rgbCamera_Kmatrix.at<float>(0, 2);
    float center_y = rgbCamera_Kmatrix.at<float>(1, 2);
    // Combine unit conversion (if necessary) with scaling by focal length for computing (X,Y)
    double unit_scaling = 1;
    float constant_x = unit_scaling / rgbCamera_Kmatrix.at<float>(0, 0);
    float constant_y = unit_scaling / rgbCamera_Kmatrix.at<float>(1, 1);

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
            pt3_rgb_target.x += std_normal_sample * ((pixel_prior.x - center_x) * constant_x);
            pt3_rgb_target.y += std_normal_sample * ((pixel_prior.y - center_y) * constant_y);
            pt3_rgb_target.z += std_normal_sample;

            depth_sq = pt3_rgb_source.z * pt3_rgb_source.z;
            std_normal_sample = distribution(generator) * (1.425e-3f) * depth_sq;
            pt3_rgb_source.x += std_normal_sample * ((pixel_frame.x - center_x) / constant_x);
            pt3_rgb_source.y += std_normal_sample * ((pixel_frame.y - center_y) / constant_y);
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
    return true;
}

bool RGBDOdometryCore::computeRelativePose(std::string& name,
        cv::Ptr<cv::FeatureDetector> detector_,
        cv::Ptr<cv::DescriptorExtractor> extractor_,
        Eigen::Matrix4f& trans,
        Eigen::Map<Eigen::Matrix<double, 6, 6> >& covMatrix,
        cv::UMat& depthimg,
        cv::UMat& frame,
        cv::Ptr<std::vector<cv::KeyPoint> >& keypoints_frame,
        cv::Ptr<cv::UMat>& descriptors_frame,
        std::vector<Eigen::Matrix4f>& transform_vector,
        float& detector_time, float& descriptor_time, float& match_time,
        float& RANSAC_time, float& covarianceTime,
        int& numFeatures, int& numMatches, int& numInliers) {
    static int bad_frames = 0;
    std::string keyframe_frameid_str = "";
    cv::UMat depth_frame;
    cv::UMat frame_vis = frame.clone(); // refresh visualization frame
    cv::UMat mask;

    // Preprocess: Convert Kinect depth image from image-of-shorts (mm) to image-of-floats (m)
    // Output: depth_frame -- a CV_32F (single float) depth image with units meters
    uchar depthtype = depthimg.getMat(cv::ACCESS_READ).type() & CV_MAT_DEPTH_MASK;
    if (depthtype == CV_16U) {
        std::cout << "Converting Kinect-style depth image to floating point depth image." << std::endl;
        int width = depthimg.cols;
        int height = depthimg.rows;
        depth_frame.create(height, width, CV_32F);
        float bad_point = std::numeric_limits<float>::quiet_NaN();
        uint16_t* uint_depthvals = (uint16_t *) depthimg.getMat(cv::ACCESS_READ).data;
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
    } else if (depthtype == CV_32F) {
        depth_frame = depthimg.clone();
    } else {
        std::cout << "Error depth frame numeric format not recognized." << std::endl;
    }

    // Preprocess: Compute mask to ignore invalid depth measurements
    // Output: mask -- 0,1 map of invalid/valid (x,y) depth measurements
    getImageFunctionProvider()->computeMask(depth_frame, mask);

    // Preprocess: Smooth or Dither the Depth measurements to reduce noise
    // Output: depth_frame -- smoothed image
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

    // Preprocess: Compute keypoint (x,y) locations and descriptors at each keypoint
    //             (x,y) location
    // Output: keypoints_frame   -- keypoint (x,y) locations
    //         descriptors_frame -- descriptor values 
    numFeatures = computeKeypointsAndDescriptors(frame, dimg, mask,
            name, detector_, extractor_,
            keypoints_frame, descriptors_frame,
            detector_time, descriptor_time, keyframe_frameid_str);

    // Preprocess: Stop execution if not enough keypoints detected
    if (keypoints_frame->size() < 10) {
        printf("Too few keypoints! Bailing on image...");
        bad_frames++;
        if (bad_frames > 2) {
            printf(" and Re-initializing the estimator.");
            prior_descriptors_.release();
        }
        return false;
    }

    // Step 1: Create a PCL point cloud object from newly detected feature points having matches/correspondence
    // Output: pcl_ptcloud_sptr -- a 3D point cloud of the 3D surface locations at all detected keypoints
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
                    dimg, rgbCamera_Kmatrix);
            //std::cout << "Added point (" << pt.x << ", " << pt.y << ", " << pt.z << ")" << std::endl;
            pcl_ptcloud_sptr->push_back(pt);
            if (std::isnan(kpt.pt.x) || std::isnan(kpt.pt.y) ||
                    std::isnan(pt.x) || std::isnan(pt.y) || std::isnan(pt.z)) {
                printf("%d : 2d (x,y)=(%f,%f)  mask(x,y)=%d (x,y,z)=(%f,%f,%f)\n",
                        i++, kpt.pt.x, kpt.pt.y,
                        mask.getMat(cv::ACCESS_READ).data[offset],
                        pt.x, pt.y, pt.z);
            }
        }
    }

    // Preprocess: Stop execution if a prior keypoints, descriptors and point cloud not available
    if (!prior_descriptors_ || prior_descriptors_->empty()) {
        prior_keypoints = keypoints_frame;
        prior_descriptors_ = descriptors_frame;
        prior_ptcloud_sptr = pcl_ptcloud_sptr;
        return false;
    }

    // Preprocess: Match (x,y) keypoints using descriptors from previous frame and
    //             descriptors from the current frame.
    // Output: good_matches -- a list of candidate correspondences between keypoints
    //                         in the prior frame and the current frame.
    match_time = 0;
    double t = (double) cv::getTickCount();
    t = (double) cv::getTickCount();
    std::vector<cv::DMatch> good_matches; // to obtain the 3D points of the model
    if (fast_match) {
        rmatcher->fastRobustMatch(good_matches, *prior_descriptors_, *descriptors_frame);
    } else {
        rmatcher->robustMatch(good_matches, *prior_descriptors_, *descriptors_frame);
    }
    //if (DEBUG) {
    std::cout << "from (" << prior_keypoints->size() << "," << keypoints_frame->size() << ")"
            << " key points found " << good_matches.size() << " good matches." << std::endl;
    //}
    // measure performance of matching algorithm
    match_time = (cv::getTickCount() - t) * 1000. / cv::getTickFrequency();

    // Preprocess: Stop execution unless enough matches exist to continue the algorithm.
    numMatches = good_matches.size();
    if (good_matches.size() < 6) {
        printf("Too few key point matches in the images! Bailing on image...");
        bad_frames++;
        if (bad_frames > 2) {
            printf(" and Re-initializing the estimator.");
            prior_descriptors_.release();
        }
        return false;
    }

    // DEBUG: code to show keypoint matches for each image pair
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
        } catch (cv::Exception& e) {
            printf("opencv exception: %s", e.what());
            return false;
        }
    }

    // Step 2: Compute 3D point cloud correspondences from the 2D feature correspondences
    //         Creates a PCL correspondence list indicating matches between points cloud points
    //         in the image pair for alignment/odometry computation
    // Output: ptcloud_matches -- a PCL Correspondences list for using in alignment algorithms
    std::vector<cv::Point2f> list_points2d_prior_match; // container for the model 2D coordinates found in the scene
    std::vector<cv::Point2f> list_points2d_scene_match; // container for the model 2D coordinates found in the scene    
    pcl::CorrespondencesPtr ptcloud_matches(new pcl::Correspondences());
    for (unsigned int match_index = 0; match_index < good_matches.size(); ++match_index) {
        cv::Point2f point2d_prior = (*prior_keypoints)[ good_matches[match_index].queryIdx ].pt; // 2D point from model
        cv::Point2f point2d_frame = (*keypoints_frame)[ good_matches[match_index].trainIdx ].pt; // 2D point from the scene
        //                list_points2d_prior_match.push_back(point2d_prior); // add 3D point
        //                list_points2d_scene_match.push_back(point2d_frame); // add 2D point
        if (false && (std::isnan(point2d_frame.x) || std::isnan(point2d_frame.y) ||
                point2d_frame.x > pcl_ptcloud_sptr->width ||
                point2d_frame.y > pcl_ptcloud_sptr->height)) {

            printf("Frame coord out of range for index pair (%d,%d) ! (x,y)=(%f,%f) (width,height)=(%d,%d)",
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

    // Step 3: Estimate the best 3D transformation using RANSAC
    // Output: trans -- the best estimate of the odometry transform
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
        printf("Too few inliers from RANSAC transform estimation! Bailing on image...");
        bad_frames++;
        if (bad_frames > 2) {
            printf(" and Re-initializing the estimator.");
            prior_descriptors_.release();
        }
        return false;
    }
    trans = ransac_rejector->getBestTransformation();
    RANSAC_time = (cv::getTickCount() - t) * 1000. / cv::getTickFrequency();
    std::cout << "RANSAC rejection left " << ptcloud_matches_ransac->size() << " inliers." << std::endl;
    numInliers = ptcloud_matches_ransac->size();
    //Eigen::toString("trans = ", trans);

    // Step 4: Estimate the covariance of our 3D transformation using the boostrap
    // Output: covMatrix -- an estimate of the transformation covariance
    estimateCovarianceBootstrap(ptcloud_matches_ransac,
            keypoints_frame,
            prior_keypoints,
            covMatrix,
            transform_vector,
            covarianceTime);

    // Post-process: Save keypoints, descriptors and point cloud of current frame
    //               as the new prior frame.
    prior_keypoints = keypoints_frame;
    prior_descriptors_ = descriptors_frame;
    prior_ptcloud_sptr = pcl_ptcloud_sptr;

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

    // Post-process: check consistency of depth frames and mask values
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
        } catch (cv::Exception& e) {
            printf("cv_bridge exception: %s", e.what());
            return true;
        }
    }
}

