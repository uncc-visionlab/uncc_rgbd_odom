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
#include <opencv2/core/types.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/eigen.hpp>

#ifdef HAVE_iGRAND
#include <rgbd_odometry/opencv_function_dev.h>
#endif

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
        Eigen::Matrix4f& transform, Eigen::Matrix<float, 6, 6> covMatrix) {
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

static std::vector<cv::Point3_<float>> reconstructParallelized(const cv::Mat_<float>& depth_image,
        const cv::Point_<float>& focal_length, const cv::Point_<float>& image_center) {

    std::vector<cv::Point3_<float>> points;
    float nan = std::numeric_limits<float>::quiet_NaN();
    points.resize(depth_image.total(), cv::Point3_<float>(nan, nan, nan));
    float width = depth_image.cols;

    depth_image.forEach(
            [&points, &width, &focal_length, &image_center](const float& z, const int* position) -> void {
                size_t pixel_y = position[0];
                size_t pixel_x = position[1];
                size_t index = pixel_y * width + pixel_x;

                if (not std::isnan(z)) {
                    float x = z * (pixel_x - image_center.x) / focal_length.x;
                            float y = z * (pixel_y - image_center.y) / focal_length.y;
                            points[index] = cv::Point3_<float>(x, y, z);
                }
            }
    );

    return points;

}

static bool inImage(const float& px, const float& py, const float& height, const float& width) {
    // checks that the pixel is within the image bounds
    return (py >= 0 and py < height and px >= 0 and px < width);
}

//bool RGBDOdometryCore::readConfigFile() {
//string filename = "I.xml";
//FileStorage fs(filename, FileStorage::WRITE);
////...
//fs.open(filename, FileStorage::READ);
//fs.release(); 
//}

bool RGBDOdometryCore::computeRelativePose(cv::UMat &frame, cv::UMat &depthimg,
        Eigen::Matrix4f& trans, Eigen::Matrix<float, 6, 6>& covMatrix) {
    float detectorTime, descriptorTime, matchTime, RANSACTime, covarianceTime;
    int numFeatures, numMatches, numInliers;

    bool odomEstimatorSuccess = computeRelativePose(frame, depthimg,
            trans, covMatrix,
            detectorTime, descriptorTime, matchTime, RANSACTime, covarianceTime,
            numFeatures, numMatches, numInliers);

    if (odomEstimatorSuccess && LOG_ODOMETRY_TO_FILE) {
        Eigen::Quaternionf quat(trans.block<3, 3>(0, 0));
        Eigen::Vector3f translation(trans.block<3, 1>(0, 3));        
        logTransformData(//keyframe_frameid_str, frame_time,
                rmatcher->detectorStr, rmatcher->descriptorStr,
                detectorTime, descriptorTime, matchTime, RANSACTime, covarianceTime,
                numFeatures, numMatches, numInliers,
                quat, translation,
                trans, covMatrix);
    }
    return odomEstimatorSuccess;
}

bool RGBDOdometryCore::computeRelativePose(cv::UMat &frameA, cv::UMat &depthimgA,
        cv::UMat &frameB, cv::UMat &depthimgB,
        Eigen::Matrix4f& trans, Eigen::Matrix<float, 6, 6>& covMatrix) {
    float detectorTime, descriptorTime, matchTime, RANSACTime, covarianceTime;
    int numFeatures, numMatches, numInliers;

    computeRelativePose(frameA, depthimgA,
            trans, covMatrix,
            detectorTime, descriptorTime, matchTime, RANSACTime, covarianceTime,
            numFeatures, numMatches, numInliers);
    bool odomEstimatorSuccess = computeRelativePose(frameB, depthimgB,
            trans, covMatrix,
            detectorTime, descriptorTime, matchTime, RANSACTime, covarianceTime,
            numFeatures, numMatches, numInliers);
    if (odomEstimatorSuccess && LOG_ODOMETRY_TO_FILE) {
        Eigen::Quaternionf quat(trans.block<3, 3>(0, 0));
        Eigen::Vector3f translation(trans.block<3, 1>(0, 3));
        logTransformData(//keyframe_frameid_str, frame_time,
                rmatcher->detectorStr, rmatcher->descriptorStr,
                detectorTime, descriptorTime, matchTime, RANSACTime, covarianceTime,
                numFeatures, numMatches, numInliers,
                quat, translation,
                trans, covMatrix);
    }
    return odomEstimatorSuccess;
}

bool RGBDOdometryCore::computeRelativePoseDirectMultiScale(
        const cv::Mat& color_img1, const cv::Mat& depth_img1, // warp image
        const cv::Mat& color_img2, const cv::Mat& depth_img2, // template image
        Eigen::Matrix4f& odometry_estimate, Eigen::Matrix<float, 6, 6>& covariance,
        int max_iterations_per_level, int start_level, int end_level) {

    bool error_decreased = false;
    bool compute_image_gradients = true;
    Eigen::Matrix4f local_odometry_estimate = odometry_estimate;
    Eigen::Matrix<float, 6, 6> local_covariance = covariance;

    std::cout << "--- Reprojection Error Minimization ---\n";

    for (int level = start_level; level >= end_level; --level) {

        std::cout << "Level: " << level << std::endl;

        int sample_factor = std::pow(2, level);

        cv::Mat sampled_depth_img1, sampled_color_img1;
        cv::resize(depth_img1, sampled_depth_img1, cv::Size(), 1.0 / sample_factor, 1.0 / sample_factor, cv::INTER_NEAREST);
        cv::resize(color_img1, sampled_color_img1, cv::Size(), 1.0 / sample_factor, 1.0 / sample_factor, cv::INTER_NEAREST);

        bool level_error_decreased = this->computeRelativePoseDirect(
                sampled_color_img1, sampled_depth_img1, color_img2, depth_img2, local_odometry_estimate, local_covariance, level, compute_image_gradients, max_iterations_per_level);

        if (level_error_decreased) {
            error_decreased = true;
            odometry_estimate = local_odometry_estimate;
            covariance = local_covariance;
        }

        if (compute_image_gradients)
            compute_image_gradients = false;

    }

    return error_decreased;

}

bool RGBDOdometryCore::computeRelativePoseDirect(
        const cv::Mat& color_img1, const cv::Mat& depth_img1, // warp image
        const cv::Mat& color_img2, const cv::Mat& depth_img2, // template image
        Eigen::Matrix4f& odometry_estimate, Eigen::Matrix<float, 6, 6>& covariance,
        int level = 0, bool compute_image_gradients = true, int max_iterations = 50) {

    // Inverse compositional image alignment with parallelization

    if (not (color_img1.isContinuous() and depth_img1.isContinuous() and color_img2.isContinuous() and depth_img2.isContinuous()))
        throw std::runtime_error("Color and Depth cv::Mats must be continuous!");

    Pose local_odometry_estimate;
    cv::Matx44f odometry_estimate_cv;
    cv::eigen2cv(odometry_estimate, odometry_estimate_cv);
    local_odometry_estimate.set(odometry_estimate_cv);
    local_odometry_estimate.invertInPlace();
    Pose delta_pose_update, prev_odometry_estimate;

    int width1 = color_img1.cols;
    int height1 = color_img1.rows;
    int width2 = color_img2.cols;
    int height2 = color_img2.rows;
    if (not this->hasRGBCameraIntrinsics()) {
        throw std::runtime_error("Camera calibration parameters are not set. Odometry cannot be estimated.");
        return false;
    }
    cv::Mat intrinsics = this->getRGBCameraIntrinsics();
    const float& fx = intrinsics.at<float>(0, 0);
    const float& fy = intrinsics.at<float>(1, 1);
    const float& cx = intrinsics.at<float>(0, 2);
    const float& cy = intrinsics.at<float>(1, 2);
    float inv_fx = 1.0f / fx;
    float inv_fy = 1.0f / fy;
    cv::Mat intensity_img1, intensity_img2;
    cv::cvtColor(color_img1, intensity_img1, cv::COLOR_RGB2GRAY);
    cv::cvtColor(color_img2, intensity_img2, cv::COLOR_RGB2GRAY);
    // normalizing intensities to range 0-1
    intensity_img1.convertTo(intensity_img1, CV_32F, 1.0 / 255.0);
    intensity_img2.convertTo(intensity_img2, CV_32F, 1.0 / 255.0);

    // compute image gradients
    static cv::Mat depth_img2_dx, depth_img2_dy, intensity_img2_dx, intensity_img2_dy;
    if (compute_image_gradients) {
        cv::Mat cdiffX = (cv::Mat_<float>(1, 3) << -1.0f, 0, 1.0f);
        cv::Mat cdiffY = (cv::Mat_<float>(3, 1) << -1.0f, 0, 1.0f);
        cv::filter2D(depth_img2, depth_img2_dx, -1, cdiffX);
        cv::filter2D(depth_img2, depth_img2_dy, -1, cdiffY);
        cv::filter2D(intensity_img2, intensity_img2_dx, -1, cdiffX);
        cv::filter2D(intensity_img2, intensity_img2_dy, -1, cdiffY);
    }

    // initialize pointclouds
    int sample_factor = std::pow(2, level);
    std::vector<cv::Point3f> points1 = reconstructParallelized(depth_img1, cv::Point2f(fx, fy) / sample_factor, cv::Point2f(cx, cy) / sample_factor);
    cv::Mat ptcloud1(points1, false);
    ptcloud1 = ptcloud1.reshape(3, height1);

    cv::Matx33f rotation;
    cv::Vec3f translation;
    cv::Mat pixels_valid(width1*height1, 1, CV_8U);
    cv::Mat depth_residuals(width1*height1, 1, CV_32F);
    cv::Mat intensity_residuals(width1*height1, 1, CV_32F);
    cv::Mat depth_gradient_vecs(width1*height1, 6, CV_32F);
    cv::Mat intensity_gradient_vecs(width1*height1, 6, CV_32F);
    cv::Mat error_hessian(6, 6, CV_32F);
    cv::Mat error_grad(6, 1, CV_32F);
    float* error_grad_ptr = error_grad.ptr<float>(0, 0);
    cv::Mat param_update(6, 1, CV_32F);
    cv::Mat error_hessian_double(6, 6, CV_64F);
    cv::Mat error_grad_double(6, 1, CV_64F);
    cv::Mat param_update_double(6, 1, CV_64F);
    float intensity_weight = 1.5;
    float intensity_weight_sq = intensity_weight*intensity_weight;

    float initial_error, error, num_constraints;
    float last_error = std::numeric_limits<float>::infinity();
    double param_max = std::numeric_limits<float>::infinity();
    bool error_decreased, enough_constraints, param_update_valid;
    std::string reason_stopped;
    int iterations = 0;
    bool iterate = true;

    while (iterate) {
        iterations++;

        // get current transformation
        rotation = local_odometry_estimate.getRotation_Matx33();
        local_odometry_estimate.getTranslation(translation);

        pixels_valid.setTo(false);

        ptcloud1.forEach<cv::Vec3f>(
                // this lambda runs for each pixel and is parallelized
                [&](const cv::Vec3f& pt, const int* position) {

                    if (not std::isnan(pt[2])) {

                        cv::Vec3f transformed_pt = rotation * pt + translation;
                        cv::Point2f warped_px;
                        warped_px.x = (transformed_pt[0] / (transformed_pt[2] * inv_fx)) + cx;
                        warped_px.y = (transformed_pt[1] / (transformed_pt[2] * inv_fy)) + cy;

                        float x0 = std::floor(warped_px.x);
                        float y0 = std::floor(warped_px.y);
                        float x1 = x0 + 1;
                        float y1 = y0 + 1;

                        if (inImage(x0, y0, height2, width2) and inImage(x1, y1, height2, width2)) {

                            // compute corner indices for pointer access
                            int row_y0 = y0*width2;
                            int row_y1 = y1*width2;
                            int index_x0y0 = row_y0 + x0;
                            int index_x0y1 = row_y1 + x0;
                            int index_x1y0 = row_y0 + x1;
                            int index_x1y1 = row_y1 + x1;

                            // compute interpolation weights
                            float x1w = warped_px.x - x0;
                            float x0w = 1.0f - x1w;
                            float y1w = warped_px.y - y0;
                            float y0w = 1.0f - y1w;

                            // interpolate depth related values
                            float depth_img2_at_warped_px = y0w * (((float *) depth_img2.data)[index_x0y0] * x0w + ((float *) depth_img2.data)[index_x1y0] * x1w) +
                                    y1w * (((float *) depth_img2.data)[index_x0y1] * x0w + ((float *) depth_img2.data)[index_x1y1] * x1w);

                            float depth2_gradx_at_warped_px = y0w * (((float *) depth_img2_dx.data)[index_x0y0] * x0w + ((float *) depth_img2_dx.data)[index_x1y0] * x1w) +
                                    y1w * (((float *) depth_img2_dx.data)[index_x0y1] * x0w + ((float *) depth_img2_dx.data)[index_x1y1] * x1w);

                            float depth2_grady_at_warped_px = y0w * (((float *) depth_img2_dy.data)[index_x0y0] * x0w + ((float *) depth_img2_dy.data)[index_x1y0] * x1w) +
                                    y1w * (((float *) depth_img2_dy.data)[index_x0y1] * x0w + ((float *) depth_img2_dy.data)[index_x1y1] * x1w);

                            if (not std::isnan(depth_img2_at_warped_px) and not std::isnan(depth2_gradx_at_warped_px) and not std::isnan(depth2_grady_at_warped_px)) {

                                // interpolate intensity related values
                                float intensity_img2_at_warped_px = y0w * (((float *) intensity_img2.data)[index_x0y0] * x0w + ((float *) intensity_img2.data)[index_x1y0] * x1w) +
                                        y1w * (((float *) intensity_img2.data)[index_x0y1] * x0w + ((float *) intensity_img2.data)[index_x1y1] * x1w);

                                float intensity2_gradx_at_warped_px = y0w * (((float *) intensity_img2_dx.data)[index_x0y0] * x0w + ((float *) intensity_img2_dx.data)[index_x1y0] * x1w) +
                                        y1w * (((float *) intensity_img2_dx.data)[index_x0y1] * x0w + ((float *) intensity_img2_dx.data)[index_x1y1] * x1w);

                                float intensity2_grady_at_warped_px = y0w * (((float *) intensity_img2_dy.data)[index_x0y0] * x0w + ((float *) intensity_img2_dy.data)[index_x1y0] * x1w) +
                                        y1w * (((float *) intensity_img2_dy.data)[index_x0y1] * x0w + ((float *) intensity_img2_dy.data)[index_x1y1] * x1w);

                                // compute index of initial point (pt)
                                int y = position[0];
                                int x = position[1];
                                int index = y * width1 + x;

                                pixels_valid.data[index] = true;

                                // compute residuals
                                float& depth_residual = ((float *) depth_residuals.data)[index];
                                depth_residual = depth_img2_at_warped_px - transformed_pt[2];

                                float& intensity_residual = ((float *) intensity_residuals.data)[index];
                                const float& intensity_img1_at_xy = ((float *) intensity_img1.data)[index];
                                intensity_residual = intensity_img2_at_warped_px - intensity_img1_at_xy;

                                // evaluate for this pixel: gradient vec = imggrad(I)*Jw at jpt
                                const cv::Vec3f& jpt = pt; // point where the jacobian will be evaluated
                                float inv_depth = 1.0f / jpt[2];
                                float inv_depth_sq = inv_depth*inv_depth;

                                float* depth_gradient_vec = depth_gradient_vecs.ptr<float>(index);
                                depth_gradient_vec[0] = depth2_gradx_at_warped_px * fx * inv_depth + depth2_grady_at_warped_px * 0;
                                depth_gradient_vec[1] = depth2_gradx_at_warped_px * 0 + depth2_grady_at_warped_px * fy * inv_depth;
                                depth_gradient_vec[2] = -(depth2_gradx_at_warped_px * fx * jpt[0] + depth2_grady_at_warped_px * fy * jpt[1]) * inv_depth_sq - 1.0f;
                                depth_gradient_vec[3] = -(depth2_gradx_at_warped_px * fx * jpt[0] * jpt[1] + depth2_grady_at_warped_px * fy * (jpt[2] * jpt[2] + jpt[1] * jpt[1])) * inv_depth_sq - jpt[1];
                                depth_gradient_vec[4] = (depth2_gradx_at_warped_px * fx * (jpt[2] * jpt[2] + jpt[0] * jpt[0]) + depth2_grady_at_warped_px * fy * jpt[0] * jpt[1]) * inv_depth_sq + jpt[0];
                                depth_gradient_vec[5] = (-depth2_gradx_at_warped_px * fx * jpt[1] + depth2_grady_at_warped_px * fy * jpt[0]) * inv_depth;

                                float* intensity_gradient_vec = intensity_gradient_vecs.ptr<float>(index);
                                intensity_gradient_vec[0] = intensity2_gradx_at_warped_px * fx * inv_depth + intensity2_grady_at_warped_px * 0;
                                intensity_gradient_vec[1] = intensity2_gradx_at_warped_px * 0 + intensity2_grady_at_warped_px * fy * inv_depth;
                                intensity_gradient_vec[2] = -(intensity2_gradx_at_warped_px * fx * jpt[0] + intensity2_grady_at_warped_px * fy * jpt[1]) * inv_depth_sq;
                                intensity_gradient_vec[3] = -(intensity2_gradx_at_warped_px * fx * jpt[0] * jpt[1] + intensity2_grady_at_warped_px * fy * (jpt[2] * jpt[2] + jpt[1] * jpt[1])) * inv_depth_sq;
                                intensity_gradient_vec[4] = (intensity2_gradx_at_warped_px * fx * (jpt[2] * jpt[2] + jpt[0] * jpt[0]) + intensity2_grady_at_warped_px * fy * jpt[0] * jpt[1]) * inv_depth_sq;
                                intensity_gradient_vec[5] = (-intensity2_gradx_at_warped_px * fx * jpt[1] + intensity2_grady_at_warped_px * fy * jpt[0]) * inv_depth;

                            }

                        }

                    }

                }

        );

        error = 0;
        num_constraints = 0;
        error_hessian.setTo(0.0f);
        error_grad.setTo(0.0f);

        // finish what we can't do in parallel
        for (int y = 0; y < height1; ++y) {
            for (int x = 0; x < width1; ++x) {

                int index = y * width1 + x;
                uchar& pixel_valid = pixels_valid.data[index];

                if (pixel_valid) {

                    float* depth_gradient_vec = depth_gradient_vecs.ptr<float>(index);
                    float* intensity_gradient_vec = intensity_gradient_vecs.ptr<float>(index);

                    // add pixel's contribution to gradient vector
                    float& depth_residual = ((float *) depth_residuals.data)[index];
                    float& intensity_residual = ((float *) intensity_residuals.data)[index];
                    for (int i = 0; i < 6; ++i) {
                        error_grad_ptr[i] += depth_residual * depth_gradient_vec[i];
                        error_grad_ptr[i] += intensity_weight * intensity_residual * intensity_gradient_vec[i];
                    }

                    // compute upper triangular component this point contributes to the Hessian
                    float* error_hessian_ptr = error_hessian.ptr<float>(0, 0);
                    for (int row = 0; row < 6; ++row) {
                        error_hessian_ptr += row;
                        for (int col = row; col < 6; ++col, ++error_hessian_ptr) {
                            *error_hessian_ptr += depth_gradient_vec[row] * depth_gradient_vec[col];
                            *error_hessian_ptr += intensity_gradient_vec[row] * intensity_gradient_vec[col];
                        }
                    }

                    error += depth_residual * depth_residual + intensity_weight_sq * intensity_residual*intensity_residual;
                    num_constraints++;

                }
            }
        }
        cv::completeSymm(error_hessian);

        if (iterations == 1)
            initial_error = error;

        error_decreased = error < last_error;
        enough_constraints = num_constraints > 6;

        if (error_decreased) {

            if (enough_constraints) {

                error_hessian.convertTo(error_hessian_double, CV_64F);
                error_grad.convertTo(error_grad_double, CV_64F);
                param_update.convertTo(param_update_double, CV_64F);

                cv::solve(error_hessian_double, error_grad_double, param_update_double);
                param_update_valid = cv::checkRange(param_update_double);

                if (not param_update_valid) { // check for NaNs
                    reason_stopped = std::string("Invalid values in parameter update!");
                } else {
                    param_update_double.convertTo(param_update, CV_32F);
                    cv::minMaxLoc(cv::abs(param_update), nullptr, &param_max);
                }

            } else {
                reason_stopped = std::string("Not enough constraints for minimization!");
            }

        } else {
            reason_stopped = std::string("Error increased.");
        }

        if (not(error_decreased and enough_constraints and param_update_valid)) {
            // don't update the parameters, stop iterating now
            local_odometry_estimate = prev_odometry_estimate;
            error = last_error;
            break;
        } else if (param_max <= 8e-6) {
            // finish this update and then stop iterating
            reason_stopped = std::string("Minimum detected.");
            iterate = false;
        } else if (iterations > max_iterations) {
            reason_stopped = std::string("Maximum iterations exceeded.");
            iterate = false;
        }

        // update parameters via composition
        prev_odometry_estimate = local_odometry_estimate;
        delta_pose_update.setFromTwist(cv::Vec3f((float *) param_update.data),
                cv::Vec3f((float *) param_update.data + 3));
        delta_pose_update.invertInPlace();
        Pose::multiply(delta_pose_update, local_odometry_estimate, local_odometry_estimate);

        last_error = error;

    }

    // invert the estimate that we return
    local_odometry_estimate.invertInPlace();
    odometry_estimate = Eigen::Map<Eigen::Matrix<float, 4, 4, Eigen::RowMajor >> (local_odometry_estimate.getTransform().val);
    covariance = Eigen::Map<Eigen::Matrix<float, 6, 6, Eigen::RowMajor >> ((float *) error_hessian.data);
    covariance = -covariance.inverse(); // Covariance matrix is the negative inverse of the Hessian
    
    std::cout << "Initial Error: " << initial_error << "\n";
    std::cout << "Final Error: " << error << "\n";
    std::cout << "Iterations: " << iterations << "\n";
    std::cout << "Reason Stopped: " << reason_stopped << "\n---\n";

    if (error <= initial_error)
        return true;
    else
        return false;

}

int RGBDOdometryCore::computeKeypointsAndDescriptors(cv::UMat& frame, cv::Mat& dimg, cv::UMat& mask,
        std::string& name,
        cv::Ptr<cv::FeatureDetector> detector_,
        cv::Ptr<cv::DescriptorExtractor> extractor_,
        cv::Ptr<std::vector<cv::KeyPoint> >& keypoints_frame,
        cv::Ptr<cv::UMat>& descriptors_frame, float& detector_time, float& descriptor_time,
        const std::string keyframe_frameid_str) {
#ifdef HAVE_iGRAND
    if (cv::iGRAND * iGRAND_detector = rmatcher->detector_.dynamicCast<cv::iGRAND>()) {
        iGRAND_detector->setDepthImage(&dimg);
    }
#endif

    double t = (double) cv::getTickCount();
    detector_->detect(frame, *keypoints_frame, mask);
    int numFeatures = keypoints_frame->size();
    detector_time = (cv::getTickCount() - t) * 1000. / cv::getTickFrequency();

    // Detect keypoint positions that, after rounding, lie within the masked region
    // and delete them from the keypoint vector.
    std::vector<cv::KeyPoint>::iterator keyptIterator;
    for (keyptIterator = keypoints_frame->begin();
            keyptIterator != keypoints_frame->end(); /*++keyptIterator*/) {
        cv::KeyPoint kpt = *keyptIterator;
        int offset = (round(kpt.pt.y) * dimg.cols + round(kpt.pt.x));
        if (((int) mask.getMat(cv::ACCESS_READ).data[offset]) == 0) {
            //float depth = ((float *) depth_frame.getMat(cv::ACCESS_READ).data)[offset];
            //                std::cout << "mask = " << ((int) mask.getMat(cv::ACCESS_READ).data[offset])
            //                        << " depth = " << depth << std::endl;
            // key point found that lies in/too close to the masked region!!
            keyptIterator = keypoints_frame->erase(keyptIterator);
            //std::cout << "called erase on pt = (" << kpt.pt.x << ", " << kpt.pt.y << ")" << std::endl;
        } else {
            ++keyptIterator;
        }
    }

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

#ifdef HAVE_iGRAND
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
        Eigen::Matrix<float, 6, 6>& covMatrix,
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
    Eigen::MatrixXf estimate_matrix(MAX_TRIALS, 6);
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
            pt3_rgb_source.x += std_normal_sample * ((pixel_frame.x - center_x) * constant_x);
            pt3_rgb_source.y += std_normal_sample * ((pixel_frame.y - center_y) * constant_y);
            pt3_rgb_source.z += std_normal_sample;

            tgtCloud.push_back(pt3_rgb_target);
            srcCloud.push_back(pt3_rgb_source);
        }
        Eigen::Matrix4f noisy_transform;
        trans_est.estimateRigidTransformation(srcCloud, tgtCloud,
                *src2tgt_correspondences, noisy_transform);
        // TODO: covariance matrix could be computed much faster by
        // directly populating matrix entries in upper diagonal
        estimate_matrix(trials, 0) = noisy_transform(0, 3);
        estimate_matrix(trials, 1) = noisy_transform(1, 3);
        estimate_matrix(trials, 2) = noisy_transform(2, 3);
        estimate_matrix(trials, 3) = noisy_transform(0, 2);
        estimate_matrix(trials, 4) = noisy_transform(1, 2);
        estimate_matrix(trials, 5) = noisy_transform(0, 1);
        //Eigen::toString("noisy_transform = ", noisy_transform);
        srcCloud.clear();
        tgtCloud.clear();
    }
    //double cov[36];
    //Eigen::Map<Eigen::Matrix<double, 6, 6> > covMatrix(cov);
    Eigen::MatrixXf centered = estimate_matrix.rowwise() - estimate_matrix.colwise().mean();
    covMatrix = (centered.adjoint() * centered) / double(estimate_matrix.rows() - 1);
    covarianceTime = (cv::getTickCount() - t) * 1000. / cv::getTickFrequency();
    return true;
}

void RGBDOdometryCore::swapOdometryBuffers() {
    //prior_image = frame.clone();
    prior_keypoints.swap(keypoints_frame);
    keypoints_frame->clear();
    prior_descriptors_.swap(descriptors_frame);
    descriptors_frame->release();
    prior_ptcloud_sptr.swap(pcl_ptcloud_sptr);
    pcl_ptcloud_sptr->clear();
}

bool RGBDOdometryCore::computeRelativePose(cv::UMat& frame, cv::UMat& depthimg,
        Eigen::Matrix4f& trans,
        Eigen::Matrix<float, 6, 6>& covMatrix,
        float& detector_time, float& descriptor_time, float& match_time,
        float& RANSAC_time, float& covarianceTime,
        int& numFeatures, int& numMatches, int& numInliers) {
    if (!hasRGBCameraIntrinsics()) {
        std::cout << "Camera calibration parameters are not set. Odometry cannot be estimated." << std::endl;
        return false;
    }
    if (!hasMatcher()) {
        std::cout << "Feature detector, descriptor and matching algorithms not set. Odometry cannot be estimated." << std::endl;
        return false;
    }
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
            rmatcher->detectorStr, rmatcher->detector_, rmatcher->extractor_,
            keypoints_frame, descriptors_frame,
            detector_time, descriptor_time, keyframe_frameid_str);

    // Preprocess: Stop execution if not enough keypoints detected
    if (keypoints_frame->size() < 10) {
        std::cout << "Too few keypoints! Bailing on image...";
        bad_frames++;
        if (bad_frames > 2) {
            std::cout << " and re-initializing the estimator." << std::endl;
            prior_image = frame.clone();
            swapOdometryBuffers();
        }
        return false;
    }

    // Step 1: Create a PCL point cloud object from newly detected feature points having matches/correspondence
    // Output: pcl_ptcloud_sptr -- a 3D point cloud of the 3D surface locations at all detected keypoints
    if (VERBOSE) {
        std::cout << "Found " << keypoints_frame->size() << " key points in frame." << std::endl;
    }
    int i = 0;
    std::vector<cv::KeyPoint>::iterator keyptIterator;
    for (keyptIterator = keypoints_frame->begin();
            keyptIterator != keypoints_frame->end(); ++keyptIterator) {
        cv::KeyPoint kpt = *keyptIterator;
        pcl::PointXYZRGB pt;
        pt = convertRGBD2XYZ(kpt.pt, frame.getMat(cv::ACCESS_READ),
                dimg, rgbCamera_Kmatrix);
        //std::cout << "Added point (" << pt.x << ", " << pt.y << ", " << pt.z << ")" << std::endl;
        pcl_ptcloud_sptr->push_back(pt);
        if (std::isnan(kpt.pt.x) || std::isnan(kpt.pt.y) ||
                std::isnan(pt.x) || std::isnan(pt.y) || std::isnan(pt.z)) {
            int offset = (round(kpt.pt.y) * dimg.cols + round(kpt.pt.x));
            printf("ERROR found NaN in 3D measurement data %d : 2d (x,y)=(%f,%f)  mask(x,y)=%d (x,y,z)=(%f,%f,%f)\n",
                    i++, kpt.pt.x, kpt.pt.y,
                    mask.getMat(cv::ACCESS_READ).data[offset],
                    pt.x, pt.y, pt.z);
        }
    }

    // Preprocess: Stop execution if prior keypoints, descriptors or point cloud not available
    if (prior_keypoints->empty()) {
        std::cout << "Aborting Odom, no prior image available." << std::endl;
        prior_image = frame.clone();
        swapOdometryBuffers();
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
    if (VERBOSE) {
        std::cout << "from (" << prior_keypoints->size() << "," << keypoints_frame->size() << ")"
                << " key points found " << good_matches.size() << " good matches." << std::endl;
    }
    // measure performance of matching algorithm
    match_time = (cv::getTickCount() - t) * 1000. / cv::getTickFrequency();

    // Preprocess: Stop execution unless enough matches exist to continue the algorithm.
    numMatches = good_matches.size();
    if (good_matches.size() < 6) {
        std::cout << "Too few key point matches in the images! Bailing on image...";
        bad_frames++;
        if (bad_frames > 2) {
            std::cout << " and re-initializing the estimator." << std::endl;
            prior_image = frame.clone();
            swapOdometryBuffers();
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

        cv::Mat ocv_depth_img_vis;
        cv::convertScaleAbs(dimg, ocv_depth_img_vis, 255.0f / 8.0f);
        cv::imshow("DEPTH", ocv_depth_img_vis);
        cv::waitKey(3);

        //std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
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
        //cv::Point2f point2d_prior = (*prior_keypoints)[ good_matches[match_index].queryIdx ].pt; // 2D point from model
        //cv::Point2f point2d_frame = (*keypoints_frame)[ good_matches[match_index].trainIdx ].pt; // 2D point from the scene
        pcl::Correspondence correspondence(good_matches[match_index].trainIdx,
                good_matches[match_index].queryIdx,
                good_matches[match_index].distance);
        ptcloud_matches->push_back(correspondence);
    }

    // Step 3: Estimate the best 3D transformation using RANSAC
    // Output: trans -- the best estimate of the odometry transform
    t = (double) cv::getTickCount();
    pcl::registration::CorrespondenceRejectorSampleConsensus<pcl::PointXYZRGB>::Ptr ransac_rejector(
            new pcl::registration::CorrespondenceRejectorSampleConsensus<pcl::PointXYZRGB>);
    ransac_rejector->setInputSource(pcl_ptcloud_sptr);
    ransac_rejector->setInputTarget(prior_ptcloud_sptr);
    ransac_rejector->setInlierThreshold(pcl_inlierThreshold);
    ransac_rejector->setRefineModel(pcl_refineModel);
    ransac_rejector->setInputCorrespondences(ptcloud_matches);
    ransac_rejector->setMaximumIterations(pcl_numIterations);
    pcl::CorrespondencesPtr ptcloud_matches_ransac(new pcl::Correspondences());
    ransac_rejector->getRemainingCorrespondences(*ptcloud_matches, *ptcloud_matches_ransac);
    if (ptcloud_matches_ransac->size() < 2) {
        std::cout << "Too few inliers from RANSAC transform estimation! Bailing on image...";
        bad_frames++;
        if (bad_frames > 2) {
            std::cout << " and re-initializing the estimator." << std::endl;
            prior_image = frame.clone();
            swapOdometryBuffers();
        }
        return false;
    }
    trans = ransac_rejector->getBestTransformation();
    RANSAC_time = (cv::getTickCount() - t) * 1000. / cv::getTickFrequency();
    if (VERBOSE) {
        std::cout << "RANSAC rejection left " << ptcloud_matches_ransac->size() << " inliers." << std::endl;
        std::cout << "trans=\n" << trans << std::endl;
    }
    numInliers = ptcloud_matches_ransac->size();

    // Step 4: Estimate the covariance of our 3D transformation using the boostrap
    // Output: covMatrix -- an estimate of the transformation covariance
    estimateCovarianceBootstrap(ptcloud_matches_ransac,
            keypoints_frame,
            prior_keypoints,
            covMatrix,
            covarianceTime);

    // Post-process: Save keypoints, descriptors and point cloud of current frame
    //               as the new prior frame.
    prior_image = frame.clone();
    swapOdometryBuffers();

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
            return false;
        }
    }
    return true;
}

