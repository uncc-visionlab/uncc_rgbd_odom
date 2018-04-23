#ifndef IMAGE_FUNCTION_DEV_H
#define	IMAGE_FUNCTION_DEV_H

// shared pointers
#include <boost/shared_ptr.hpp>

// OpenCV includes
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/ocl.hpp>

#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

class ImageFunctionProvider {
public:
    typedef boost::shared_ptr<ImageFunctionProvider> Ptr;

    void extractImageSpaceParameters();

    cv::ocl::Context getOpenCLContext() {
        return context;
    };

    void setOpenCLContext(cv::ocl::Context cxt) {
        context = cxt;
    };

    int initOpenCL();

    int initialize(bool useOpenCL, std::string opencl_path, std::string progname);

    void setOpenCLPath(std::string path) {
        opencl_path = path;
        opencl_path += (*opencl_path.rbegin() != '/') ? "/" : "";
    };

    std::string getOpenCLPath() {
        return opencl_path;
    };

    void setDepthmaskOpenCL(std::string file) {
        depthmask_cl = file;
    }

    std::string getDepthMaskOpenCL() {
        return opencl_path + depthmask_cl;
    }

    int init_ocl_computeMask();

    void computeMask(cv::InputArray src, cv::OutputArray dst, int code = -1,
            std::string keyframe_frameid_str = "debug");
    bool _ocl_computeMask(cv::InputArray src, cv::OutputArray dst, int code = -1);
    void _cpu_computeMask(cv::Mat src, cv::Mat dst, int code = -1);

    void infsToValue(cv::InputArray src, cv::OutputArray dst, float value = 0,
            std::string keyframe_frameid_str = "debug");
    bool _ocl_infsToValue(cv::InputArray src, cv::OutputArray dst, float value);
    void _cpu_infsToValue(cv::Mat src, cv::Mat dst, float value);

    void nansToValue(cv::InputArray src, cv::OutputArray dst, float value = 0,
            std::string keyframe_frameid_str = "debug");
    bool _ocl_nansToValue(cv::InputArray src, cv::OutputArray dst, float value);
    void _cpu_nansToValue(cv::Mat src, cv::Mat dst, float value);

    //void centerAndRescale(cv::InputArray src, cv::OutputArray dst, 
    //        float max, float mean, std::vector<int> indices,
    //        std::string keyframe_frameid_str = "debug");

    template<typename T>
    void centerAndRescale(cv::InputArray src,
            cv::OutputArray dst, T max, T mean, std::vector<int> indices,
            std::string keyframe_frameid_str) {
        if (!dst.sameSize(src)) {
            dst.create(src.size(), src.type());
        }
        if (cv::ocl::useOpenCL() && dst.isUMat() && _ocl_centerAndRescale(src, dst, max, mean, indices)) {
            //        cv::imwrite(keyframe_frameid_str + "_opencl_mask.png", dst);
            return;
        }
        cv::Mat src_cpu = src.getMat();
        cv::Mat dst_cpu = dst.getMat();
        _cpu_centerAndRescale<T>(src_cpu, dst_cpu, max, mean, indices);
        //    cv::imwrite(keyframe_frameid_str + "_cpu_mask.png", dst_cpu);
    }

    //bool _ocl_centerAndRescale(cv::InputArray src, cv::OutputArray dst, 
    //float max, float mean, std::vector<int> indices);

    bool _ocl_centerAndRescale(cv::InputArray src, cv::OutputArray dst, float max, float mean, std::vector<int> indices) {
        return false;
    }

    //void _cpu_centerAndRescale(cv::Mat src, cv::Mat dst, 
    //float max, float mean, std::vector<int> indices);

    template<typename T>
    void _cpu_centerAndRescale(cv::Mat src, cv::Mat dst,
            T maxV, T mean, std::vector<int> indices) {
        T* srcv = (T *) src.data;
        T* dstv = (T *) dst.data;
        std::vector<int>::iterator idxIter = indices.begin();
        int nextValidIdx = *idxIter;
        for (int idx = 0; idx < src.rows * src.cols; ++idx) {
            if (idx != nextValidIdx) {
                dstv[idx] = mean;
            } else {
                dstv[idx] = srcv[idx];
                idxIter++;
                nextValidIdx = *idxIter;
            }
        }
        float min = std::numeric_limits<T>::max(), max = -std::numeric_limits<T>::max();
        for (std::vector<int>::iterator idxIter = indices.begin();
                idxIter != indices.end(); ++idxIter) {
            if (srcv[*idxIter] > max) {
                max = srcv[*idxIter];
            }
            if (srcv[*idxIter] < min) {
                min = srcv[*idxIter];
            }
        }
        T scale = maxV / (max - min);
        for (std::vector<int>::iterator idxIter = indices.begin();
                idxIter != indices.end(); ++idxIter) {
            dstv[*idxIter] = (srcv[*idxIter] - min) * scale + mean;
        }
    }


    //void sobel(cv::InputArray src, cv::OutputArray dst, 
    //int dx_order, int dy_order, std::vector<int> indices,
    //        std::string keyframe_frameid_str = "debug");

    template<typename typeIn, typename typeOut>
    void sobel(cv::InputArray src,
            cv::OutputArray dst, int dx_order, int dy_order,
            std::vector<int> indices,
            std::string keyframe_frameid_str) {
        if (!dst.sameSize(src)) {
            dst.create(src.size(), src.type());
        }
        if (cv::ocl::useOpenCL() && dst.isUMat() && _ocl_sobel(src, dst, dx_order, dy_order, indices)) {
            //        cv::imwrite(keyframe_frameid_str + "_opencl_mask.png", dst);
            return;
        }
        cv::Mat src_cpu = src.getMat();
        cv::Mat dst_cpu = dst.getMat();
        _cpu_sobel<typeIn, typeOut>(src_cpu, dst_cpu, dx_order, dy_order, indices);
        //    cv::imwrite(keyframe_frameid_str + "_cpu_mask.png", dst_cpu);
    }

    bool _ocl_sobel(cv::InputArray src, cv::OutputArray dst,
            int dx_order, int dy_order, std::vector<int> indices) {
        return false;
    }

    template<typename typeIn, typename typeOut>
    void _cpu_sobel(cv::Mat src, cv::Mat dst,
            int dx_order, int dy_order, std::vector<int> indices) {
        typeIn* srcv = (typeIn *) src.data;
        typeOut* dstv = (typeOut *) dst.data;
        int w = src.cols;
        int h = src.rows;
        int idx, prow, nrow;
        if (dy_order == 1) {
            for (std::vector<int>::iterator idxIter = indices.begin();
                    idxIter != indices.end(); ++idxIter) {
                idx = *idxIter;
                prow = idx - w;
                nrow = idx + w;

                dstv[idx] = -static_cast<typeOut> (srcv[prow - 1])
                        - 2 * static_cast<typeOut> (srcv[prow])
                        - static_cast<typeOut> (srcv[prow + 1])
                        + static_cast<typeOut> (srcv[nrow - 1])
                        + 2 * static_cast<typeOut> (srcv[nrow])
                        + static_cast<typeOut> (srcv[nrow + 1]);
            }
        } else if (dx_order == 1) {
            for (std::vector<int>::iterator idxIter = indices.begin();
                    idxIter != indices.end(); ++idxIter) {
                idx = *idxIter;
                prow = idx - w;
                nrow = idx + w;
                dstv[idx] = -static_cast<typeOut> (srcv[prow - 1])
                        - 2 * static_cast<typeOut> (srcv[idx - 1])
                        - static_cast<typeOut> (srcv[nrow - 1])
                        + static_cast<typeOut> (srcv[prow + 1])
                        + 2 * static_cast<typeOut> (srcv[idx + 1])
                        + static_cast<typeOut> (srcv[nrow + 1]);
            }
        }
    }

    void _cpu_ditherDepthAndSmooth(cv::Mat src, cv::Mat dst,
            std::string keyframe_frameid_str);

    void ditherDepthAndSmooth(cv::InputArray src, cv::OutputArray dst,
            std::string keyframe_frameid_str);
    // banks of Gaussian filters
    void freeFilterBank();
    void computeFilterBank();
    void _cpu_depthFilter(cv::Mat src, cv::Mat dst);
    void depthFilter(cv::InputArray src, cv::OutputArray dst,
            std::string keyframe_frameid_str);
    // moving average filter
    void movingAvgFilter(cv::InputArray src, cv::OutputArray dst,
            std::string keyframe_frameid_str);
    void _cpu_movingAvgFilter(cv::Mat src, cv::Mat dst,
            std::string keyframe_frameid_str);
protected:
    std::string opencl_path, depthmask_cl;
    cv::ocl::Context context;
    //    cv::ocl::Kernel kernel;
    cv::ocl::ProgramSource oclsrc;
    cv::String compile_flags;
};

#endif	/* IMAGE_FUNCTION_DEV_H */

