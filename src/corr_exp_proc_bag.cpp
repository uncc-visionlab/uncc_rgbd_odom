/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
#include <ros/ros.h>
#include <ros/time.h>
#include <std_msgs/String.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>

// OpenCV includes
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

// ROS Bridge to OpenCV
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/core/core.hpp>

#define DEBUG false
#define MAX_NUM_BAGFILES 18
#define NUM_SAMPLE_FUNCTIONS 100

static int outputIndex = 0;

void writeNextRGBDMessage(rosbag::View::iterator view_iter,
        rosbag::View::iterator view_iter_end,
        std::string outputfile, ros::Time& myTime,
        int sampleIndex) {
    bool haveRGBImage = false, haveDepthImage = false, haveCameraInfo = false;
    sensor_msgs::Image rgb_image;
    sensor_msgs::Image depth_image;
    sensor_msgs::CameraInfo camera_info;
    bool stop = false;
    int currentIndex = 0;
    rosbag::Bag output_bag;
    if (DEBUG) {
        std::cout << "Opening " << outputfile << " for appending new data...." << std::endl;
    }
    output_bag.open(outputfile, rosbag::bagmode::Append);

    for (; view_iter != view_iter_end && !stop; ++view_iter) {
        rosbag::MessageInstance m = *view_iter;

        sensor_msgs::Image::ConstPtr img = m.instantiate<sensor_msgs::Image>();
        if (img != NULL) {
            if (DEBUG) {
                std::cout << "Heard an image message on topic " << m.getTopic()
                        << " Time = " << m.getTime() << "." << std::endl;
            }
            if (m.getTopic().find("depth") != std::string::npos) {
                // THIS IS A DEPTH IMAGE MESSAGE
                depth_image = *img;
                haveDepthImage = true;
            }
            if (m.getTopic().find("rgb") != std::string::npos) {
                // THIS IS A RGB IMAGE MESSAGE
                rgb_image = *img;
                haveRGBImage = true;
                if (sampleIndex == 5 && sampleIndex == currentIndex) {
                    cv_bridge::CvImageConstPtr cv_rgbimg_ptr = cv_bridge::toCvShare(img, sensor_msgs::image_encodings::BGR8);
                    cv::UMat frame = cv_rgbimg_ptr->image.getUMat(cv::ACCESS_READ);
                    cv::imwrite("bagfile_" + std::to_string(++outputIndex) + "_image.png", frame);
                }
            }
        }
        sensor_msgs::CameraInfo::ConstPtr ci = m.instantiate<sensor_msgs::CameraInfo>();
        if (ci != NULL) {
            if (DEBUG) {
                std::cout << "Heard an camera info message on topic " << m.getTopic()
                        << " Time = " << m.getTime() << "." << std::endl;
            }
            // THIS IS A CAMERA INFO MESSAGE
            camera_info = *ci;
            haveCameraInfo = true;
        }
        if (haveRGBImage && haveDepthImage && haveCameraInfo) {
            haveRGBImage = haveDepthImage = haveCameraInfo = false;
            if (DEBUG) {
                std::cout << "sampleIndex = " << sampleIndex << " currentIndex = " << currentIndex << std::endl;
            }
            if (sampleIndex == currentIndex) {
                std::cout << "Writing new RGBD image data messages to output bag at time index "
                        << myTime << std::endl;
                depth_image.header.stamp = myTime;
                output_bag.write("/camera/depth_registered/input_image", myTime, depth_image);
                myTime += ros::Duration(0.001); // 1 msec between depth & rgb image messages
                rgb_image.header.stamp = myTime;
                output_bag.write("/camera/rgb/input_image", myTime, rgb_image);
                myTime += ros::Duration(0.001); // 1 msec between rgb & camera info messages
                camera_info.header.stamp = myTime;
                output_bag.write("/camera/rgb/camera_info", myTime, camera_info);
                stop = true;
            }
            currentIndex++;
        }
    }
    output_bag.close();
}

int main(int argc, char **argv) {
    rosbag::Bag input_bag[MAX_NUM_BAGFILES];
    rosbag::View::iterator input_view_iter[MAX_NUM_BAGFILES];
    rosbag::View::iterator input_view_iter_end[MAX_NUM_BAGFILES];

    int numBagfiles = argc - 1;
    if (numBagfiles < 1) {
        std::cout << "Please specify the input bag file as an argument." << std::endl;
        return 0;
    }

    std::vector<std::string> topics;
    topics.push_back(std::string("/camera/rgb/input_image"));
    topics.push_back(std::string("/camera/rgb/camera_info"));
    topics.push_back(std::string("/camera/depth_registered/input_image"));

    ros::Time::init();

    std::cout << "Multiplexing " << numBagfiles << " bag files." << std::endl;
    for (int bagIdx = 0; bagIdx < numBagfiles; ++bagIdx) {
        std::cout << "Opening bag file " << (bagIdx + 1)
                << ": " << argv[bagIdx + 1] << " for reading ...." << std::endl;
        input_bag[bagIdx].open(argv[bagIdx + 1], rosbag::bagmode::Read);
        rosbag::View* viewptr = new rosbag::View(input_bag[bagIdx], rosbag::TopicQuery(topics));
        input_view_iter[bagIdx] = viewptr->begin();
        input_view_iter_end[bagIdx] = viewptr->end();
    }

    rosbag::Bag output_bag;
    std::string outputfile = "output.bag";
    std::cout << "Opening " << outputfile << " for writing...." << std::endl;
    output_bag.open(outputfile, rosbag::bagmode::Write);
    std::cout << "Initializing output file " << outputfile << "." << std::endl;
    std_msgs::String str;
    str.data = std::string("foo");
    output_bag.write("chatter", ros::Time::now(), str);
    output_bag.close();

    ros::Time myTime = ros::Time::now();
    for (int sampleFunctionIndex = 0; sampleFunctionIndex < NUM_SAMPLE_FUNCTIONS;
            ++sampleFunctionIndex) {
        for (int bagIdx = 0; bagIdx < numBagfiles; ++bagIdx) {
            std::cout << "Operating on bag file " << (bagIdx + 1)
                    << " retrieving message at index " << sampleFunctionIndex << "...." << std::endl;
            writeNextRGBDMessage(input_view_iter[bagIdx],
                    input_view_iter_end[bagIdx],
                    outputfile, myTime, sampleFunctionIndex);
            myTime += ros::Duration(0.1); // specifies sec between RGBD image messages
        }
    }

    for (int bagIdx = 0; bagIdx < numBagfiles; ++bagIdx) {
        std::cout << "Closing bag file " << (bagIdx + 1)
                << ": " << argv[bagIdx + 1] << " for reading ...." << std::endl;
        input_bag[bagIdx].close();
    }
}