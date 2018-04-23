/* 
 * File:   RobustMatcher.h
 * Author: arwillis
 *
 * Created on August 18, 2015, 2:36 PM
 */

#ifndef ROBUSTMATCHER_H
#define ROBUSTMATCHER_H

#include <iostream>
#include <map>

#include <boost/shared_ptr.hpp>

#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include <opencv2/flann/flann.hpp>
#include <opencv2/flann/dist.h>

#define HAVE_iGRAND

#ifdef HAVE_iGRAND
#include <rgbd_odometry/opencv_function_dev.h>
#endif

#define MAX_KEYPOINTS 1000

class RobustMatcher {
public:
    typedef boost::shared_ptr<RobustMatcher> Ptr;

    RobustMatcher(std::string detector_name = "ORB",
            std::string descriptor_name = "ORB",
            int nkeypoints = 600) :
    ratio_(0.8f),
    num_keypoints(nkeypoints) {
#ifdef HAVE_iGRAND
        factory["iGRAND"] = create_iGRAND; // ORB is the default feature
#endif
        factory["ORB"] = create_ORB; // ORB is the default feature
        factory["SIFT"] = create_SIFT;
        factory["SURF"] = create_SURF;
        factory["BRISK"] = create_BRISK;
        //factory["MSER"] = create_MSER;   // not working
        //factory["KAZE"] = create_KAZE;   // not working
        //factory["AKAZE"] = create_AKAZE; // not working  
        factory["BRIEF"] = create_BRIEF; // only a descriptor extractor
        factory["GFTT"] = create_GFTT; // only a detector
        factory["FAST"] = create_FAST; // only a detector
        //factory["FREAK"] = create_FREAK; // only an extractor, not working
        //factory["STAR"] = create_STAR;   // only a detector, not working


        setFeatureDetector(detector_name);
        setDescriptorExtractor(descriptor_name);
        // Probably you have tried to use KD - Tree or KMeans ? They works only for
        // CV_32F descriptors like SIFT or SURF. For binary descriptors like BRIEF\ORB\FREAK
        // you have to use either LSH or Hierarchical clustering index. Or simple bruteforce search.
        // You can manage it automatically, for example like this.
        //cv::flann::Index tree = GenFLANNIndex(descriptors);
        // BruteForce matcher with Norm Hamming is the default matcher
        //matcher_ = new cv::BFMatcher((int) cv::NORM_HAMMING, false);
    }

    virtual ~RobustMatcher();

    // Set the feature detector

    void setFeatureDetector(std::string detector_name) {
        detectorStr = detector_name;
        cv::Ptr<cv::FeatureDetector> new_detector_ =
                factory[detector_name]().dynamicCast<cv::FeatureDetector>();
        if (new_detector_) {
            detector_ = new_detector_;
        } else {
            std::cerr << "Could not create feature detector " << detector_name
                    << "!" << std::endl;
        }
    }

    // Set the descriptor extractor

    void setDescriptorExtractor(std::string extractor_name) {
        descriptorStr = extractor_name;        
        cv::Ptr<cv::DescriptorExtractor> new_extractor_ =
                factory[extractor_name]().dynamicCast<cv::DescriptorExtractor>();
        if (new_extractor_) {
            extractor_ = new_extractor_;
            matcher_ = createMatcher(extractor_);
        } else {
            std::cerr << "Could not create feature descriptor extractor "
                    << extractor_name << "!" << std::endl;
        }
    }

    // Set the matcher

    void setDescriptorMatcher(const cv::Ptr<cv::DescriptorMatcher>& match) {
        matcher_ = match;
    }

    // Set ratio parameter for the ratio test

    void setRatio(float rat) {
        ratio_ = rat;
    }

    // Clear matches for which NN ratio is > than threshold
    // return the number of removed points
    // (corresponding entries being cleared,
    // i.e. size will be 0)
    int ratioTest(std::vector<std::vector<cv::DMatch> > &matches);

    // Insert symmetrical matches in symMatches vector
    void symmetryTest(const std::vector<std::vector<cv::DMatch> >& matches1,
            const std::vector<std::vector<cv::DMatch> >& matches2,
            std::vector<cv::DMatch>& symMatches);

    // Match feature points using ratio and symmetry test
    void robustMatch(std::vector<cv::DMatch>& good_matches,
            const cv::UMat& descriptors_frame,
            const cv::UMat& descriptors_model);

    // Match feature points using ratio test
    void fastRobustMatch(std::vector<cv::DMatch>& good_matches,
            const cv::UMat& descriptors_frame,
            const cv::UMat& descriptors_model);

    // pointer to the feature point detector object
    cv::Ptr<cv::FeatureDetector> detector_;
    // pointer to the feature descriptor extractor object
    cv::Ptr<cv::DescriptorExtractor> extractor_;
    std::string detectorStr;
    std::string descriptorStr;

private:
    // pointer to the matcher object
    cv::Ptr<cv::DescriptorMatcher> matcher_;
    // max ratio between 1st and 2nd NN
    float ratio_;

    int num_keypoints;
    typedef cv::Ptr<cv::Algorithm> (*algo_creator_t)();
    typedef cv::Ptr<cv::DescriptorMatcher> (*algo_matcher_t)();
    std::map<std::string, algo_creator_t> factory;

    // proxy functions are needed since SIFT::create() etc. have optional parameters,
    // so the function pointers can not be unified.

#ifdef HAVE_iGRAND
    static cv::Ptr<cv::Algorithm> create_iGRAND() {
        cv::Ptr<cv::Algorithm> algo_ptr = cv::iGRAND::create(MAX_KEYPOINTS);
        return algo_ptr;
    };
#endif
    
    static cv::Ptr<cv::Algorithm> create_ORB() {
        //        return cv::ORB::create();
        cv::Ptr<cv::Algorithm> algo_ptr = cv::ORB::create(MAX_KEYPOINTS);
        //        printParams(algo_ptr);
        //        algo_ptr->set("maxCorners", 600);
        return algo_ptr;
    };

    static cv::Ptr<cv::Algorithm> create_SIFT() {
        return cv::xfeatures2d::SIFT::create(MAX_KEYPOINTS);
    };

    static cv::Ptr<cv::Algorithm> create_SURF() {
        return cv::xfeatures2d::SURF::create();
    };

    static cv::Ptr<cv::Algorithm> create_BRISK() {
        return cv::BRISK::create();
    };

    static cv::Ptr<cv::Algorithm> create_FAST() {
        return cv::FastFeatureDetector::create();
        //        return cv::FeatureDetector::create("GridFAST");
    };

    static cv::Ptr<cv::Algorithm> create_GFTT() {
        return cv::GFTTDetector::create(MAX_KEYPOINTS);
    };

    static cv::Ptr<cv::Algorithm> create_BRIEF() {
        return cv::xfeatures2d::BriefDescriptorExtractor::create();
    };

    // below: not working
    //    static cv::Ptr<cv::Algorithm> create_STAR() {
    //        //return cv::FeatureDetector.create("STAR");
    //        //StarDetectorParams paramStar = cvStarDetectorParams();
    //        //paramStar.responseThreshold *= 2.4;
    //        //paramStar.suppressNonmaxSize = 15;
    //        //paramStar.lineThresholdProjected *=5;
    //        //paramStar.lineThresholdBinarized *=5;
    //        return cv::StarFeatureDetector();
    //        return cv::FeatureDetector.create("STAR");
    //    };

    static cv::Ptr<cv::Algorithm> create_MSER() {
        return cv::MSER::create();
    };

    static cv::Ptr<cv::Algorithm> create_FREAK() {
        return cv::xfeatures2d::FREAK::create();
    };

    static cv::Ptr<cv::Algorithm> create_KAZE() {
        return cv::KAZE::create();
    };

    static cv::Ptr<cv::Algorithm> create_AKAZE() {
        return cv::AKAZE::create();
    };

    // below: untested

    static cv::Ptr<cv::Algorithm> create_AGAST() {
        return cv::AgastFeatureDetector::create();
    };

    static cv::Ptr<cv::Algorithm> create_LUCID() {
        return cv::xfeatures2d::LUCID::create(3, 3);
    };

    static cv::Ptr<cv::Algorithm> create_LATCH() {
        return cv::xfeatures2d::LATCH::create();
    };

    static cv::Ptr<cv::Algorithm> create_DAISY() {
        return cv::xfeatures2d::DAISY::create();
    };

    static cv::Ptr<cv::DescriptorMatcher> createMatcher(cv::Ptr<cv::Feature2D> algo) {
        //typedef cv::flann::L2<unsigned char> Distance_U8;
        //cv::flann::GenericIndex< Distance_U8>* m_flann;

        // FLANN - KDTree
        // trees = 5
        cv::Ptr<cv::flann::IndexParams> index_params_32f = new cv::flann::KDTreeIndexParams(4);
        //index_params.setAlgorithm(cvflann::FLANN_INDEX_KDTREE);
        // int checks = 32, float eps = 0, bool sorted = true
        cv::Ptr<cv::flann::SearchParams> search_params_32f = new cv::flann::SearchParams(32, 0, true);

        // Fast Library Approximate Nearest Neighbor (FLANN) - LSH
        //                   table_number = 6, //# 12
        //                   key_size = 12,     //# 20
        //                   multi_probe_level = 1) //#2
        cv::Ptr<cv::flann::IndexParams> index_params_8u = new cv::flann::LshIndexParams(6, 12, 1); //(20,10,2)
        //cv::Ptr<cv::flann::IndexParams> index_params = new cv::flann::HierarchicalClusteringIndexParams();
        // int checks = 32, float eps = 0, bool sorted = true
        cv::Ptr<cv::flann::SearchParams> search_params_8u = new cv::flann::SearchParams(32, 0, true);
        std::string type;
        switch (algo->descriptorType()) {
            case CV_32F:
                std::cout << "Descriptor is type CV_32F" << std::endl;
                // Brute Force
                //return cv::makePtr<cv::BFMatcher>((int) cv::NORM_L2, false);
                //return cv::makePtr<cv::BFMatcher>((int) cv::NORM_L2SQR, false);
                return cv::makePtr<cv::FlannBasedMatcher>(index_params_32f, search_params_32f);
            case CV_8U:
                type = "CV_8U";
                //            case CV_16U:
                //                type = "CV_16U";
                std::cout << "Descriptor is type " << type << std::endl;
            default:
                // Brute Force
                //return cv::makePtr<cv::BFMatcher>((int) cv::NORM_HAMMING, false);
                // Fast Library Approximate Nearest Neighbor (FLANN))
                return cv::makePtr<cv::FlannBasedMatcher>(index_params_8u, search_params_8u);
        }
    };

    cv::flann::Index GenFLANNIndex(cv::Mat keys) {
        switch (keys.type()) {
            case CV_32F:
            {
                return cv::flann::Index(keys, cv::flann::KDTreeIndexParams(4));
                break;
            }
            case CV_8U:
                //            case CV_16U:
            {
                cvflann::flann_distance_t dist_type = cvflann::FLANN_DIST_HAMMING;
                //                cvflann::flann_distance_t dist_type = cvflann::FLANN_DIST_L2;
                return cv::flann::Index(keys, cv::flann::HierarchicalClusteringIndexParams(),
                        dist_type);
                break;
            }
            default:
            {
                return cv::flann::Index(keys, cv::flann::KDTreeIndexParams(4));
                break;
            }
        }
    }

    //    void printParams(cv::Ptr<cv::Algorithm> algorithm) {
    //        std::vector<std::string> parameters;
    //        algorithm->getParams(parameters);
    //
    //        for (int i = 0; i < (int) parameters.size(); i++) {
    //            std::string param = parameters[i];
    //            int type = algorithm->paramType(param);
    //            std::string helpText = algorithm->paramHelp(param);
    //            std::string typeText;
    //
    //            switch (type) {
    //                case cv::Param::BOOLEAN:
    //                    typeText = "bool";
    //                    break;
    //                case cv::Param::INT:
    //                    typeText = "int";
    //                    break;
    //                case cv::Param::REAL:
    //                    typeText = "real (double)";
    //                    break;
    //                case cv::Param::STRING:
    //                    typeText = "string";
    //                    break;
    //                case cv::Param::MAT:
    //                    typeText = "Mat";
    //                    break;
    //                case cv::Param::ALGORITHM:
    //                    typeText = "Algorithm";
    //                    break;
    //                case cv::Param::MAT_VECTOR:
    //                    typeText = "Mat vector";
    //                    break;
    //            }
    //            std::cout << "Parameter '" << param << "' type=" << typeText << " help=" << helpText << std::endl;
    //        }
    //    }
};

#endif /* ROBUSTMATCHER_H */

