// Standard C++ includes
#include <string>
#include <iostream>
#include <fstream>

// ROS includes
#include <ros/ros.h>
//#include <cv_bridge/cv_bridge.h>

// OpenCV includes
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// Includes for this Library
#include <rgbd_odometry/image_function_dev.h>

#define IMAGE_MASK_MARGIN 20

void ImageFunctionProvider::extractImageSpaceParameters() {
    //
    //
    //    cv::UMat mask; // type of mask is CV_8U      
    //    computeMask(depth_frame, mask);
    //    cv::Mat maskvals = mask.getMat(cv::ACCESS_READ);
    //    depth_img2->setMask(mask);    
    //    cv::UMat laplacian, dIdx_umat, dIdy_umat, dIdx_int_umat, dIdy_int_umat;
    //    int kernelsize = 1;
    //    //cv::Laplacian(depth_frame, laplacian, CV_32F, kernelsize);
    //    //    nansToZeros(depth_frame, depth_frame);
    //    //infsToValue(depth_frame, depth_frame, avgDepth);
    //    //cv::Sobel(depth_frame, dIdx_umat, CV_32F, 1, 0, kernelsize);
    //    //cv::Sobel(depth_frame, dIdy_umat, CV_32F, 0, 1, kernelsize);
    //    sobel(depth_frame, dIdx_umat, 1, 0, indices);
    //    sobel(depth_frame, dIdy_umat, 0, 1, indices);
    //
    //    //infsToValue(depth_frame, depth_frame, 0);
    //    //    nansToZeros(depth_frame, depth_frame);
    //    centerAndRescale(depth_frame, depth_frame, 255, 20, indices);
    //    //    nansToZeros(dIdx_umat, dIdx_umat);
    //    centerAndRescale(dIdx_umat, dIdx_umat, 255, 20, indices);
    //    //    nansToZeros(dIdy_umat, dIdy_umat);
    //    centerAndRescale(dIdy_umat, dIdy_umat, 255, 20, indices);
    //    try {
    //        cv::imwrite(keyframe_frameid_str + ".png", frame);
    //        cv::imwrite(keyframe_frameid_str + "_depth.png", depth_frame);
    //        cv::imwrite(keyframe_frameid_str + "_dIdx.png", dIdx_umat);
    //        cv::imwrite(keyframe_frameid_str + "_dIdy.png", dIdy_umat);
    //    } catch (cv_bridge::Exception& e) {
    //        ROS_ERROR("cv_bridge exception: %s", e.what());
    //        return;
    //    }
}

int ImageFunctionProvider::initialize(bool useOpenCL, std::string opencl_path,
        std::string progfilename) {
    if (cv::ocl::haveOpenCL()) {
        cv::ocl::setUseOpenCL(useOpenCL); // disable OpenCL in the processing of UMat  
        if (cv::ocl::useOpenCL()) {
            setOpenCLPath(opencl_path);
            setDepthmaskOpenCL(progfilename);
            if (initOpenCL() == EXIT_FAILURE) {
                std::cout << "Failed initializing OpenCL!" << std::endl;
                return (EXIT_FAILURE);
            }
        } else {
            std::cout << "OpenCL is AVAILABLE but is NOT ENABLED!" << std::endl;
        }
    } else {
        std::cout << "OpenCL is NOT AVAILABLE!" << std::endl;
    }
}

int ImageFunctionProvider::initOpenCL() {
    std::cout << "OpenCL is AVAILABLE and ENABLED!" << std::endl;
    if (!context.create(cv::ocl::Device::TYPE_GPU)) {
        std::cout << "Failed creating the context..." << std::endl;
        return (EXIT_FAILURE);
    }

    // In OpenCV 3.0.0 beta, only a single device is detected.
    std::cout << context.ndevices() << " GPU devices are detected." << std::endl;
    for (int i = 0; i < context.ndevices(); i++) {
        cv::ocl::Device device = context.device(i);
        std::cout << "name                 : " << device.name() << std::endl;
        std::cout << "available            : " << device.available() << std::endl;
        std::cout << "imageSupport         : " << device.imageSupport() << std::endl;
        std::cout << "OpenCL_C_Version     : " << device.OpenCL_C_Version() << std::endl;
        std::cout << std::endl;
    }
    return init_ocl_computeMask();
}

int ImageFunctionProvider::init_ocl_computeMask() {
    // Read the OpenCL kernel code
    std::string depthmask_src_fullpath = getDepthMaskOpenCL();
    std::ifstream ifs(depthmask_src_fullpath.c_str());
    if (ifs.fail()) {
        std::cout << "Depthmask OpenCL source code not found at path: " << depthmask_src_fullpath << std::endl;
        return (EXIT_FAILURE);
    }

    std::string kernelSource((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    cv::ocl::ProgramSource programSource_loc(kernelSource);
    oclsrc = programSource_loc;
    //    std::cout << oclsrc.source() << std::endl;
    // Set the compilation flags for the kernel code
    compile_flags = cv::format("-DMARGIN=%d", IMAGE_MASK_MARGIN); // "-D dstT=float"

    return (EXIT_SUCCESS);
}

bool ImageFunctionProvider::_ocl_computeMask(cv::InputArray src, cv::OutputArray dst, int code) {
    // get the kernel; kernel is compiled only once and subsequently cached
    cv::ocl::Kernel kernel("depthmaskf", oclsrc, compile_flags);
    kernel.args(cv::ocl::KernelArg::ReadOnlyNoSize(src.getUMat()),
            cv::ocl::KernelArg::ReadWrite(dst.getUMat()));
    //    kernel.args(src, dst);    
    size_t globalThreads[3] = {(long unsigned int)src.cols(), (long unsigned int)src.rows(), 1};
    //size_t localThreads[3] = { 16, 16, 1 };
    bool success = kernel.run(3, globalThreads, NULL, true);
    if (!success) {
        std::cout << "Failed running the kernel..." << std::endl;
    }
    return success;
}

void ImageFunctionProvider::_cpu_computeMask(cv::Mat src, cv::Mat dst, int code) {
    float* srcv;
    uchar* dstv;
    static uchar BACKGROUND = cv::GC_BGD;
    static uchar FOREGROUND = 255 * cv::GC_FGD;
    for (int row = 0; row < src.rows; ++row) {
        srcv = (float *) &src.data[row * src.cols * sizeof (float)];
        dstv = (uchar *) & dst.data[row * dst.cols * sizeof (uchar)];
        for (int col = 0; col < src.cols; ++col) {
            if (std::isnan(srcv[col]) ||
                    (col < IMAGE_MASK_MARGIN) || (row < IMAGE_MASK_MARGIN) ||
                    (src.cols - col < IMAGE_MASK_MARGIN) ||
                    (src.rows - row < IMAGE_MASK_MARGIN)) {
                dstv[col] = BACKGROUND;
            } else {
                dstv[col] = FOREGROUND;
            }
        }
    }
}

void ImageFunctionProvider::computeMask(cv::InputArray src, cv::OutputArray dst, int code,
        std::string keyframe_frameid_str) {
    if (!dst.sameSize(src)) {
        dst.create(src.size(), CV_8U);
    }
    if (cv::ocl::useOpenCL() && dst.isUMat() && _ocl_computeMask(src, dst, code)) {
        //        cv::imwrite(keyframe_frameid_str + "_opencl_mask.png", dst);
        return;
    }
    cv::Mat src_cpu = src.getMat();
    cv::Mat dst_cpu = dst.getMat();
    _cpu_computeMask(src_cpu, dst_cpu, code);
    //    cv::imwrite(keyframe_frameid_str + "_cpu_mask.png", dst_cpu);
}

bool ImageFunctionProvider::_ocl_infsToValue(cv::InputArray src, cv::OutputArray dst, float value) {
    return false;
}

void ImageFunctionProvider::_cpu_infsToValue(cv::Mat src, cv::Mat dst, float value) {
    float* srcv;
    float* dstv;
    for (int row = 0; row < src.rows; ++row) {
        srcv = (float *) &src.data[row * src.cols * sizeof (float)];
        dstv = (float *) &dst.data[row * dst.cols * sizeof (float)];
        for (int col = 0; col < src.cols; ++col) {
            if (std::isinf(srcv[col])) {
                dstv[col] = value;
            } else {
                dstv[col] = srcv[col];
            }
        }
    }
}

void ImageFunctionProvider::infsToValue(cv::InputArray src, cv::OutputArray dst, float value,
        std::string keyframe_frameid_str) {
    if (!dst.sameSize(src)) {
        dst.create(src.size(), src.type());
    }
    if (cv::ocl::useOpenCL() && dst.isUMat() && _ocl_infsToValue(src, dst, value)) {
        //        cv::imwrite(keyframe_frameid_str + "_opencl_mask.png", dst);
        return;
    }
    cv::Mat src_cpu = src.getMat();
    cv::Mat dst_cpu = dst.getMat();
    _cpu_infsToValue(src_cpu, dst_cpu, value);
    //    cv::imwrite(keyframe_frameid_str + "_cpu_mask.png", dst_cpu);
}

bool ImageFunctionProvider::_ocl_nansToValue(cv::InputArray src, cv::OutputArray dst, float value) {
    return false;
}

void ImageFunctionProvider::_cpu_nansToValue(cv::Mat src, cv::Mat dst, float value) {
    float* srcv;
    float* dstv;
    for (int row = 0; row < src.rows; ++row) {
        srcv = (float *) &src.data[row * src.cols * sizeof (float)];
        dstv = (float *) &dst.data[row * dst.cols * sizeof (float)];
        for (int col = 0; col < src.cols; ++col) {
            if (std::isnan(srcv[col])) {
                dstv[col] = value;
            } else {
                dstv[col] = srcv[col];
            }
        }
    }
}

void ImageFunctionProvider::nansToValue(cv::InputArray src, cv::OutputArray dst, float value,
        std::string keyframe_frameid_str) {
    if (!dst.sameSize(src)) {
        dst.create(src.size(), src.type());
    }
    if (cv::ocl::useOpenCL() && dst.isUMat() && _ocl_nansToValue(src, dst, value)) {
        //        cv::imwrite(keyframe_frameid_str + "_opencl_mask.png", dst);
        return;
    }
    cv::Mat src_cpu = src.getMat();
    cv::Mat dst_cpu = dst.getMat();
    _cpu_nansToValue(src_cpu, dst_cpu, value);
    //    cv::imwrite(keyframe_frameid_str + "_cpu_mask.png", dst_cpu);
}

void ImageFunctionProvider::_cpu_ditherDepthAndSmooth(cv::Mat src, cv::Mat dst,
        std::string keyframe_frameid_str) {
    int width = src.cols;
    int height = src.rows;
    int prev_knot_x = -1;
    int x_runlength = 0;
    int y_runlength[width];
    int prev_knot_y[width];
    for (int i = 0; i < width; ++i) {
        prev_knot_y[i] = -1;
        y_runlength[i] = 0;
    }
    cv::Mat gradXm(src.size(), CV_32F, float(0));
    cv::Mat gradYm(src.size(), CV_32F, float(0));
    cv::Mat ditherm;
    ditherm = src.clone();
    cv::Mat discontinuity_mapm(src.size(), CV_8U, cv::Scalar(0));
    //cv::Mat discontinuity_mapm(src.size(), CV_32F, float(0));
    float *depth = (float *) src.data;
    float *idepth = (float *) dst.data;
    float *gradX = (float *) gradXm.data;
    float *gradY = (float *) gradYm.data;
    float *dither = (float *) ditherm.data;
    unsigned char *discontinuity_map = (unsigned char *) discontinuity_mapm.data;
    //float *discontinuity_map = (float *) discontinuity_mapm.data;
    float quant_step, noise, l1_dist;
    int radius, max_L1dist, rand_dist, d2z_dx2, d2z_dy2, tmp;
    bool boundary = false;
    int offset = 0;
    for (int y = 0; y < height; ++y, offset += width) {
        for (int x = 0; x < width; ++x) {
            // compute gradient
            if (!std::isfinite(depth[offset + x]) ||
                    !std::isfinite(depth[offset + x + 1]) ||
                    x == width - 1 || y == height - 1) {
                continue;
            }

            radius = (int) ((depth[offset + x] - 0.6f) / (0.6f));
            if (radius <= 1 || y <= radius || height - y <= radius ||
                    x <= radius || width - x <= radius) {
                continue;
            }
            gradX[offset + x] = depth[offset + x + 1] - depth[offset + x];
            gradY[offset + x] = depth[offset + width + x] - depth[offset + x];
            quant_step = 2.85e-3 * depth[offset + x] * depth[offset + x];
            max_L1dist = 2 * radius;
            if (gradX[offset + x] == 0 && gradY[offset + x] == 0) { // Gradient is zero
                // Laplacian smoothing
                d2z_dx2 = depth[offset + x] - 0.5 * (depth[offset + x - radius] + depth[offset + x + radius]);

                if (fabs(d2z_dx2) < 1.5 * quant_step) {
                    dither[offset + x] = dither[offset + x] - d2z_dx2;
                } else {
                    discontinuity_map[offset + x] = 1;
                }
                d2z_dy2 = depth[offset + x] - 0.5 * (depth[offset - (radius * width) + x] +
                        depth[offset + (radius * width) + x]);
                if (fabs(d2z_dy2) < 1.5 * quant_step) {
                    dither[offset + x] = dither[offset + x] - d2z_dy2;
                } else {
                    discontinuity_map[offset + x] = 1;
                }

            } else { // Gradient is non-zero

                if (x > 1 && discontinuity_map[offset + x - 1] == 1) {
                    discontinuity_map[offset + x] = 1;
                } 
                if (fabs(gradX[offset + x]) < 2 * quant_step &&
                        fabs(gradY[offset + x]) < 2 * quant_step) {
                    for (int win_y = -(radius - 1); win_y <= radius; ++win_y) {
                        for (int win_x = -(radius - 1); win_x <= radius; ++win_x) {
                            if (!std::isfinite(depth[(y + win_y) * width + win_x]) ||
                                    !std::isfinite(depth[(y - win_y) * width - win_x])) {
                                continue;
                            }
                            l1_dist = fabs(win_x) + fabs(win_y);
                            if (win_x <= 0) l1_dist++;
                            if (win_y <= 0) l1_dist++;
                            rand_dist = 1.2 * max_L1dist * ((float) rand()) / RAND_MAX;
                            if (rand_dist > l1_dist) {
                                if (fabs(dither[(y + win_y) * width + x + win_x] -
                                        dither[(y - win_y) * width + x - win_x]) < 2.5 * quant_step) {
                                    tmp = dither[(y + win_y) * width + x + win_x];
                                    dither[(y + win_y) * width + x + win_x] = dither[(y - win_y) * width + x - win_x];
                                    dither[(y - win_y) * width + x - win_x] = tmp;
                                }
                            }
                        }
                    } // end dither
                    if (gradX[offset + x] != 0 && gradY[offset + x] != 0) {
                        dither[offset + x] = dither[offset + x] - 0.5 * (gradX[offset + x] + gradY[offset + x]);
                    } else if (gradX[offset + x] != 0) {
                        dither[offset + x] = dither[offset + x] - 0.5 * gradX[offset + x];
                    } else if (gradY[offset + x] != 0) {
                        dither[offset + x] = dither[offset + x] - 0.5 * gradY[offset + x];
                    } // end quantization boundary smoothing
                } // smooth surface criterion
            } // end stanza for Gradient != 0
        }
    }
    float sumval, numvals;
    offset = 0;
    for (int y = 0; y < height; ++y, offset += width) {
        for (int x = 0; x < width; ++x) {
            idepth[offset + x] = depth[offset+x];
            if (!std::isfinite(depth[offset + x])) {
                continue;
            }

            radius = (int) ((depth[offset + x] - 0.6f) / (0.6f));
            if (radius <= 1 || y <= radius || height - y <= radius ||
                    x <= radius || width - x <= radius) {
                continue;
            }
            quant_step = 2.85e-3 * depth[offset + x] * depth[offset + x];
            sumval = 0;
            numvals = 0;
            idepth[offset + x] = 0;
            for (int win_y = y - radius; win_y <= y + radius; ++win_y) {
                for (int win_x = x - radius; win_x <= x + radius; ++win_x) {
                    if (std::isfinite(depth[win_y * width + win_x]) &&
                            discontinuity_map[win_y * width + win_x] == 0) {
                        sumval = sumval + dither[win_y*width + win_x];
                        numvals++;
                    }
                }
            }
            if (numvals > 0) {
                idepth[offset + x] = sumval / numvals;
            } else {
                idepth[offset+x] = 0;
            }
        }
    }

    try {
//        cv::imwrite(keyframe_frameid_str + "_discontinuities.png", discontinuity_mapm*200);
//        cv::imwrite(keyframe_frameid_str + "_depth.png", 5*(src+10));
//        cv::imwrite(keyframe_frameid_str + "_dither.png", ditherm);
//        cv::imwrite(keyframe_frameid_str + "_idepth.png", 5*(dst + 10));
        //        cv::imwrite(keyframe_frameid_str + "_dIdy.png", dIdy_umat);
    } catch (cv::Exception& e) {
        printf("cv_bridge exception: %s", e.what());
        return;
    }
}

void ImageFunctionProvider::ditherDepthAndSmooth(cv::InputArray src, cv::OutputArray dst,
        std::string keyframe_frameid_str) {
    if (!dst.sameSize(src)) {
        dst.create(src.size(), CV_32F);
    }
    cv::Mat src_cpu = src.getMat();
    cv::Mat dst_cpu = dst.getMat();
    _cpu_ditherDepthAndSmooth(src_cpu, dst_cpu, keyframe_frameid_str);
}

float** filterBank;

void ImageFunctionProvider::freeFilterBank() {
    for (int i = 0; i < 7; ++i)
        delete [] filterBank[i];
    delete [] filterBank;
}

void ImageFunctionProvider::computeFilterBank() {
    float mean_depth[] = {1.60012, 2.87916, 4.11713, 4.98310};
    int NUM_FILTERS = 4;
    filterBank = new float*[NUM_FILTERS];
    float k;
    float sigma_xy;
    float filtersum;
    for (int filterIdx = 0; filterIdx < NUM_FILTERS; ++filterIdx) {
        int windowdim = (2 * (filterIdx + 1) + 1);
        int halfdim = windowdim >> 1;
        filterBank[filterIdx] = new float[windowdim * windowdim];
        //        sigma_xy = mean_depth[filterIdx]*(1.0f / 4.0f)*(5.0f / 3.0f);
        sigma_xy = windowdim;
        k = 1.0f / (sigma_xy * sigma_xy * 2.0f * M_PI);
        filtersum = 0;
        int offset = 0;
        for (int y = -halfdim; y <= halfdim; ++y) {
            for (int x = -halfdim; x <= halfdim; ++x, ++offset) {
                filterBank[filterIdx][offset] = k * exp(-(x * x + y * y) / (2.0f * sigma_xy * sigma_xy));
                filtersum += filterBank[filterIdx][offset];
            }
        }
        //std::cout << "Computing window size " << windowdim << " filter" << std::endl;
        for (int y = 0; y < windowdim; ++y) {
            for (int x = 0; x < windowdim; ++x) {
                filterBank[filterIdx][y * windowdim + x] /= filtersum;
                //std::cout << filterBank[filterIdx][y * windowdim + x] << " ";
            }
            //std::cout << std::endl;
        }
    }
}

void ImageFunctionProvider::_cpu_depthFilter(cv::Mat src, cv::Mat dst) {
    float* srcv;
    float* dstv;
    float quantization_step;
    int filterIdx, windowdim, halfdim, row_offset, dst_offset, filt_offset, filtered;
    srcv = (float *) src.data;
    dstv = (float *) dst.data;
    filtered = 0;
    for (int row = 0; row < src.rows; ++row) {
        for (int col = 0; col < src.cols; ++col) {
            dst_offset = row * src.cols + col;
            dstv[dst_offset] = srcv[dst_offset];
            if (!std::isnan(srcv[dst_offset])) {
                if ((col > IMAGE_MASK_MARGIN) || (row > IMAGE_MASK_MARGIN) ||
                        (src.cols - col > IMAGE_MASK_MARGIN) ||
                        (src.rows - row > IMAGE_MASK_MARGIN)) {
                    filterIdx = (int) (round(srcv[dst_offset]*5.0f / 6.0f) - 1);
                    quantization_step = 0.00285f * srcv[dst_offset] * srcv[dst_offset];
                    if (filterIdx > 3) {
                        filterIdx = 3;
                    }
                    if (filterIdx >= 0) {
                        windowdim = (2 * (filterIdx + 1) + 1);
                        halfdim = windowdim >> 1;
                        filt_offset = 0;
                        dstv[dst_offset] = 0;
                        for (int y = -halfdim; y <= halfdim; ++y) {
                            row_offset = y * src.cols;
                            for (int x = -halfdim; x <= halfdim; ++x, ++filt_offset) {
                                if (std::isnan(srcv[dst_offset + row_offset + x])) {
                                    dstv[dst_offset] = srcv[dst_offset];
                                    goto stop_filtering;
                                }
                                if (fabs(srcv[dst_offset + row_offset + x] - srcv[dst_offset]) > 2.0f * quantization_step) {
                                    //                                    std::cout << "depth = " << srcv[dst_offset] << " "
                                    //                                            << fabs(srcv[dst_offset + row_offset + x] - srcv[dst_offset])
                                    //                                            << " > " << 2.0f * quantization_step << std::endl;
                                    dstv[dst_offset] = srcv[dst_offset];
                                    goto stop_filtering;
                                }
                                dstv[dst_offset] += srcv[dst_offset + row_offset + x] *
                                        filterBank[filterIdx][filt_offset];
                            }
                        }
                        filtered++;
stop_filtering:
                        if (fabs(dstv[dst_offset] - srcv[dst_offset]) > 2.0f * quantization_step) {
                            std::cout << "original depth = " << srcv[dst_offset] << " "
                                    << "new depth = " << dstv[dst_offset] << std::endl;
                            dstv[dst_offset] = srcv[dst_offset];
                            dstv[dst_offset] = srcv[dst_offset];
                            filtered--;
                        }
                    }
                }
            }
        }
    }
    std::cout << "Filtered " << filtered << " depth values." << std::endl;
}

void ImageFunctionProvider::depthFilter(cv::InputArray src, cv::OutputArray dst,
        std::string keyframe_frameid_str) {
    dst.create(src.size(), CV_32F);
    //if (cv::ocl::useOpenCL() && dst.isUMat() && _ocl_computeMask(src, dst, code)) {
    //        cv::imwrite(keyframe_frameid_str + "_opencl_mask.png", dst);
    //return;
    //}
    cv::Mat src_cpu = src.getMat();
    cv::Mat dst_cpu = dst.getMat();
    _cpu_depthFilter(src_cpu, dst_cpu);
    cv::imwrite(keyframe_frameid_str + "_depthfilter.png", dst_cpu);
}


void ImageFunctionProvider::_cpu_movingAvgFilter(cv::Mat src, cv::Mat dst,
        std::string keyframe_frameid_str) {
    int width = src.cols;
    int height = src.rows;
    //cv::Mat discontinuity_mapm(src.size(), CV_8U, cv::Scalar(0));
    //unsigned char *discontinuity_map = (unsigned char *) discontinuity_mapm.data;
    float *depth = (float *) src.data;
    float *idepth = (float *) dst.data;
    int radius, offset = 0;
    float sumval, numvals;
    offset = 0;
    for (int y = 0; y < height; ++y, offset += width) {
        for (int x = 0; x < width; ++x) {
            idepth[offset + x] = depth[offset+x];
            if (!std::isfinite(depth[offset + x])) {
                continue;
            }

            radius = (int) ((depth[offset + x] - 0.6f) / (0.6f));
            if (radius <= 1 || y <= radius || height - y <= radius ||
                    x <= radius || width - x <= radius) {
                continue;
            }
            sumval = 0;
            numvals = 0;
            idepth[offset + x] = 0;
            for (int win_y = y - radius; win_y <= y + radius; ++win_y) {
                for (int win_x = x - radius; win_x <= x + radius; ++win_x) {
                    if (std::isfinite(depth[win_y * width + win_x])){
//                        &&
//                            discontinuity_map[win_y * width + win_x] == 0) {
                        sumval = sumval + depth[win_y*width + win_x];
                        numvals++;
                    }
                }
            }
            if (numvals > 0) {
                idepth[offset + x] = sumval / numvals;
            } else {
                idepth[offset+x] = 0;
            }
        }
    }

    try {
//        cv::imwrite(keyframe_frameid_str + "_discontinuities.png", discontinuity_mapm*200);
//        cv::imwrite(keyframe_frameid_str + "_depth.png", 5*(src+10));
//        cv::imwrite(keyframe_frameid_str + "_dither.png", ditherm);
//        cv::imwrite(keyframe_frameid_str + "_idepth.png", 5*(dst + 10));
        //        cv::imwrite(keyframe_frameid_str + "_dIdy.png", dIdy_umat);
    } catch (cv::Exception& e) {
        printf("cv_bridge exception: %s", e.what());
        return;
    }
}


void ImageFunctionProvider::movingAvgFilter(cv::InputArray src, cv::OutputArray dst,
        std::string keyframe_frameid_str) {
    dst.create(src.size(), CV_32F);
    //if (cv::ocl::useOpenCL() && dst.isUMat() && _ocl_computeMask(src, dst, code)) {
    //        cv::imwrite(keyframe_frameid_str + "_opencl_mask.png", dst);
    //return;
    //}
    cv::Mat src_cpu = src.getMat();
    cv::Mat dst_cpu = dst.getMat();
    _cpu_depthFilter(src_cpu, dst_cpu);
    cv::imwrite(keyframe_frameid_str + "_depthfilter.png", dst_cpu);
}
