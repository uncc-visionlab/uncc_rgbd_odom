#ifndef PROBABILITY_AND_STATISTICS_FUNCTION_DEV_H
#define	PROBABILITY_AND_STATISTICS_FUNCTION_DEV_H

// shared pointers
#include <boost/shared_ptr.hpp>

// OpenCV includes
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/ocl.hpp>

#ifdef OPENCV3
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#else
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#endif

#include <pcl/registration/correspondence_rejection.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>

#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_registration.h>

#include <pcl/common/transforms.h>

namespace pcl {
    namespace registration {

        /** \brief CorrespondenceRejectorSampleConsensus implements a correspondence rejection
         * using Random Sample Consensus to identify inliers (and reject outliers)
         * \author Dirk Holz
         * \ingroup registration
         */
        template <typename PointT>
        class CorrespondenceRejectorSampleConsensusWithCovariance : public CorrespondenceRejectorSampleConsensus<PointT> {
            //////////////////////////////////////////////////////////////////////////

//            void
//            getRemainingCorrespondences(
//                    const pcl::Correspondences& original_correspondences,
//                    pcl::Correspondences& remaining_correspondences) {
//                if (!input_) {
//                    PCL_ERROR("[pcl::registration::%s::getRemainingCorrespondences] No input cloud dataset was given!\n", getClassName().c_str());
//                    return;
//                }
//
//                if (!target_) {
//                    PCL_ERROR("[pcl::registration::%s::getRemainingCorrespondences] No input target dataset was given!\n", getClassName().c_str());
//                    return;
//                }
//
//                if (save_inliers_)
//                    inlier_indices_.clear();
//
//                int nr_correspondences = static_cast<int> (original_correspondences.size());
//                std::vector<int> source_indices(nr_correspondences);
//                std::vector<int> target_indices(nr_correspondences);
//
//                // Copy the query-match indices
//                for (size_t i = 0; i < original_correspondences.size(); ++i) {
//                    source_indices[i] = original_correspondences[i].index_query;
//                    target_indices[i] = original_correspondences[i].index_match;
//                }
//
//                // from pcl/registration/icp.hpp:
//                std::vector<int> source_indices_good;
//                std::vector<int> target_indices_good;
//                {
//                    // From the set of correspondences found, attempt to remove outliers
//                    // Create the registration model
//                    typedef typename pcl::SampleConsensusModelRegistration<PointT>::Ptr SampleConsensusModelRegistrationPtr;
//                    SampleConsensusModelRegistrationPtr model;
//                    model.reset(new pcl::SampleConsensusModelRegistration<PointT> (input_, source_indices));
//                    // Pass the target_indices
//                    model->setInputTarget(target_, target_indices);
//                    // Create a RANSAC model
//                    pcl::RandomSampleConsensus<PointT> sac(model, inlier_threshold_);
//                    sac.setMaxIterations(max_iterations_);
//
//                    // Compute the set of inliers
//                    if (!sac.computeModel()) {
//                        remaining_correspondences = original_correspondences;
//                        best_transformation_.setIdentity();
//                        return;
//                    } else {
//                        if (refine_ && !sac.refineModel()) {
//                            PCL_ERROR("[pcl::registration::CorrespondenceRejectorSampleConsensus::getRemainingCorrespondences] Could not refine the model! Returning an empty solution.\n");
//                            return;
//                        }
//
//                        std::vector<int> inliers;
//                        sac.getInliers(inliers);
//
//                        if (inliers.size() < 3) {
//                            remaining_correspondences = original_correspondences;
//                            best_transformation_.setIdentity();
//                            return;
//                        }
//                        boost::unordered_map<int, int> index_to_correspondence;
//                        for (int i = 0; i < nr_correspondences; ++i)
//                            index_to_correspondence[original_correspondences[i].index_query] = i;
//
//                        remaining_correspondences.resize(inliers.size());
//                        for (size_t i = 0; i < inliers.size(); ++i)
//                            remaining_correspondences[i] = original_correspondences[index_to_correspondence[inliers[i]]];
//
//                        if (save_inliers_) {
//                            inlier_indices_.reserve(inliers.size());
//                            for (size_t i = 0; i < inliers.size(); ++i)
//                                inlier_indices_.push_back(index_to_correspondence[inliers[i]]);
//                        }
//
//                        // get best transformation
//                        Eigen::VectorXf model_coefficients;
//                        sac.getModelCoefficients(model_coefficients);
//                        best_transformation_.row(0) = model_coefficients.segment<4>(0);
//                        best_transformation_.row(1) = model_coefficients.segment<4>(4);
//                        best_transformation_.row(2) = model_coefficients.segment<4>(8);
//                        best_transformation_.row(3) = model_coefficients.segment<4>(12);
//                    }
//                }
//            }

            //template <typename PointT> 
            //            bool pcl::RandomSampleConsensus<PointT>::computeModel(int) {
            //                // Warn and exit if no threshold was set
            //                if (threshold_ == std::numeric_limits<double>::max()) {
            //                    PCL_ERROR("[pcl::RandomSampleConsensus::computeModel] No threshold set!\n");
            //                    return (false);
            //                }
            //
            //                iterations_ = 0;
            //                int n_best_inliers_count = -INT_MAX;
            //                double k = 1.0;
            //
            //                std::vector<int> selection;
            //                Eigen::VectorXf model_coefficients;
            //
            //                double log_probability = log(1.0 - probability_);
            //                double one_over_indices = 1.0 / static_cast<double> (sac_model_->getIndices()->size());
            //
            //                int n_inliers_count = 0;
            //                unsigned skipped_count = 0;
            //                // supress infinite loops by just allowing 10 x maximum allowed iterations for invalid model parameters!
            //                const unsigned max_skip = max_iterations_ * 10;
            //
            //                // Iterate
            //                while (iterations_ < k && skipped_count < max_skip) {
            //                    // Get X samples which satisfy the model criteria
            //                    sac_model_->getSamples(iterations_, selection);
            //
            //                    if (selection.empty()) {
            //                        PCL_ERROR("[pcl::RandomSampleConsensus::computeModel] No samples could be selected!\n");
            //                        break;
            //                    }
            //
            //                    // Search for inliers in the point cloud for the current plane model M
            //                    if (!sac_model_->computeModelCoefficients(selection, model_coefficients)) {
            //                        //++iterations_;
            //                        ++skipped_count;
            //                        continue;
            //                    }
            //
            //                    // Select the inliers that are within threshold_ from the model
            //                    //sac_model_->selectWithinDistance (model_coefficients, threshold_, inliers);
            //                    //if (inliers.empty () && k > 1.0)
            //                    //  continue;
            //
            //                    n_inliers_count = sac_model_->countWithinDistance(model_coefficients, threshold_);
            //
            //                    // Better match ?
            //                    if (n_inliers_count > n_best_inliers_count) {
            //                        n_best_inliers_count = n_inliers_count;
            //
            //                        // Save the current model/inlier/coefficients selection as being the best so far
            //                        model_ = selection;
            //                        model_coefficients_ = model_coefficients;
            //
            //                        // Compute the k parameter (k=log(z)/log(1-w^n))
            //                        double w = static_cast<double> (n_best_inliers_count) * one_over_indices;
            //                        double p_no_outliers = 1.0 - pow(w, static_cast<double> (selection.size()));
            //                        p_no_outliers = (std::max) (std::numeric_limits<double>::epsilon(), p_no_outliers); // Avoid division by -Inf
            //                        p_no_outliers = (std::min) (1.0 - std::numeric_limits<double>::epsilon(), p_no_outliers); // Avoid division by 0.
            //                        k = log_probability / log(p_no_outliers);
            //                    }
            //
            //                    ++iterations_;
            //                    PCL_DEBUG("[pcl::RandomSampleConsensus::computeModel] Trial %d out of %f: %d inliers (best is: %d so far).\n", iterations_, k, n_inliers_count, n_best_inliers_count);
            //                    if (iterations_ > max_iterations_) {
            //                        PCL_DEBUG("[pcl::RandomSampleConsensus::computeModel] RANSAC reached the maximum number of trials.\n");
            //                        break;
            //                    }
            //                }
            //
            //                PCL_DEBUG("[pcl::RandomSampleConsensus::computeModel] Model: %lu size, %d inliers.\n", model_.size(), n_best_inliers_count);
            //
            //                if (model_.empty()) {
            //                    inliers_.clear();
            //                    return (false);
            //                }
            //
            //                // Get the set of inliers that correspond to the best model found so far
            //                sac_model_->selectWithinDistance(model_coefficients_, threshold_, inliers_);
            //                return (true);
            //            }


        };
    }
}

class ProbabilityAndStatisticsFunctionProvider {
public:
    typedef boost::shared_ptr<ProbabilityAndStatisticsFunctionProvider> Ptr;
    void computePoseCovariance(float *x, float *y, float *z, float *covMat);
protected:
};

#endif	/* PROBABILITY_AND_STATISTICS_FUNCTION_DEV_H */

