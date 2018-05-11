/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   pose.hpp
 * Author: arwillis
 *
 * Created on October 10, 2017, 10:55 AM
 */

#ifndef POSE_HPP
#define POSE_HPP

#ifdef __cplusplus

#include <boost/make_shared.hpp>
#include <boost/shared_ptr.hpp>

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>

class Pose {
    
private:
    static cv::Matx33f computeMatrixV(const cv::Vec3f& rotation) {
        // Computes the matrix V used in the exponential map
        
        float theta = cv::norm(rotation);
        if (theta > 1e-7) {
            cv::Matx33f w_cross(0.0, -rotation(2), rotation(1), rotation(2), 0.0, -rotation(0), -rotation(1), rotation(0), 0.0);
            cv::Matx33f w_cross_sq = rotation*rotation.t();
            return (cv::Matx33f::eye() + ((1-cos(theta))/pow(theta, 2))*w_cross + ((theta - sin(theta))/pow(theta, 3))*w_cross_sq);
        } else {
            return (cv::Matx33f::eye());
        }
        
    }
    
    static cv::Matx33f computeMatrixInvV(const cv::Vec3f& rotation) {
        // Computes the closed form inverse of the matrix V used in the log map
        
        float theta = cv::norm(rotation);
        if (theta > 1e-7) {
            cv::Matx33f w_cross(0.0, -rotation(2), rotation(1), rotation(2), 0.0, -rotation(0), -rotation(1), rotation(0), 0.0);
            cv::Matx33f w_cross_sq = rotation*rotation.t();
            return (cv::Matx33f::eye() - 0.5*w_cross + (1/pow(theta, 2))*((1 - (theta*sin(theta)))/(2*(1-cos(theta))))*w_cross_sq);
        } else {
            return (cv::Matx33f::eye());
        }
        
    }
    
public:

    Pose() : rodrigues(0, 0, 0), position(0, 0, 0) {

    }

    Pose(cv::Vec3f _position, cv::Vec3f _rodrigues = cv::Vec3f(0, 0, 0)) {
        position = _position;
        rodrigues = _rodrigues;
    }

    virtual ~Pose() {
    }
    
    void rotateInPlace(cv::Vec3f& vec) {
        cv::Mat rotMat;
        cv::Rodrigues(rodrigues, rotMat);
        float *mm = rotMat.ptr<float>(0, 0);
        vec[0] = mm[0] * vec[0] + mm[1] * vec[1] + mm[2] * vec[2];
        vec[1] = mm[3] * vec[0] + mm[4] * vec[1] + mm[5] * vec[2];
        vec[2] = mm[6] * vec[0] + mm[7] * vec[1] + mm[8] * vec[2];
    }

    void transformInPlace(cv::Vec3f& pt) {
        cv::Mat rotMat;
        cv::Rodrigues(rodrigues, rotMat);
        //std::cout << "rodMat " << rotMat << std::endl;
        float *mm = rotMat.ptr<float>(0, 0);
        pt[0] = mm[0] * pt[0] + mm[1] * pt[1] + mm[2] * pt[2] + position[0];
        pt[1] = mm[3] * pt[0] + mm[4] * pt[1] + mm[5] * pt[2] + position[1];
        pt[2] = mm[6] * pt[0] + mm[7] * pt[1] + mm[8] * pt[2] + position[2];
    }
    
    void getTranslation(cv::Vec3f& _position) const {
        _position = position;
    }
    
    cv::Mat getRotation_Mat() const {
        cv::Mat rotMat;
        cv::Rodrigues(rodrigues, rotMat);
        return rotMat;
    }

    cv::Matx33f getRotation_Matx33() const {
        cv::Mat rotMat;
        cv::Rodrigues(rodrigues, rotMat);
        return cvtMat_to_Matx33(rotMat);
    }

    static cv::Matx33f cvtMat_to_Matx33(const cv::Mat& rotMat) {
        cv::Matx33f rotMatx;
        const float *rotmat = rotMat.ptr<float>(0, 0);
        for (int i = 0; i < 9; ++i) {
            rotMatx.val[i] = *rotmat++;
        }
        return rotMatx;
    }

    void set(const cv::Matx44f tform) {
        cv::Mat rotMat(3, 3, CV_32F);
        float *rotmat = rotMat.ptr<float>(0, 0);
        for (int i = 0, j = 3; i < 11; ++i) {
            *rotmat++ = tform.val[i];
            if ((i + 1) % j == 0) {
                j += 4;
                i++;
            }
        }
        cv::Vec3f _rodrigues;
        cv::Rodrigues(rotMat, _rodrigues);
        this->rodrigues = _rodrigues;
        this->position[0] = tform.val[3];
        this->position[1] = tform.val[7];
        this->position[2] = tform.val[11];
    }

    void set(cv::Vec3f _position, cv::Vec3f _rodrigues) {
        this->rodrigues = _rodrigues;
        this->position = _position;
    }
    
    void setFromTwist(const cv::Vec3f& translation, const cv::Vec3f _rodrigues) {
        // Sets this to the transformation corresponding to the SE3 twist (translation, rodrigues)
        this->rodrigues =  _rodrigues;
        this->position = this->computeMatrixV(this->rodrigues)*translation;
    }

    void get(cv::Vec3f& _position, cv::Vec3f& _rodrigues) const {
        _rodrigues = this->rodrigues;
        _position = this->position;
    }

    void set(cv::Vec3f _position, cv::Mat _rotMat) {
        this->position = _position;
        cv::Rodrigues(_rotMat, this->rodrigues);
    }

    void setPosition(cv::Vec3f _position) {
        this->position = _position;
    }

    void getTransform(cv::Matx44f& tform) {
        cv::Mat rotMat = getRotation_Mat();
        float *rotmat = rotMat.ptr<float>(0, 0);
        for (int i = 0, j = 3; i < 11; ++i) {
            tform.val[i] = *rotmat++;
            if ((i + 1) % j == 0) {
                j += 4;
                i++;
            }
        }
        tform.val[3] = this->position[0];
        tform.val[7] = this->position[1];
        tform.val[11] = this->position[2];
    }

    static void multiply(Pose &transformA, Pose &transformB, Pose &dest) {
        cv::Matx44f xformA = transformA.getTransform();
        cv::Matx44f xformB = transformB.getTransform();
        //std::cout << "xform1 = " << xform1 << std::endl;
        //std::cout << "xform2 = " << xform2 << std::endl;
        //std::cout << "xform1*xform2 = " << xform1*xform2 << std::endl;
        dest.set(xformA * xformB);
    }
    
    Pose operator * (Pose transform) const {
        Pose result;
        result.set(this->getTransform()*transform.getTransform());
        return result;
    }
    
    void invertInPlace() {
        cv::Matx44f xformA = getTransform();
        this->set(xformA.inv());
    }

    cv::Matx44f getTransform() const {
        static cv::Mat rotMat;
        cv::Matx44f tform;
        cv::Rodrigues(rodrigues, rotMat);
        float *rotmat = rotMat.ptr<float>(0, 0);
        for (int i = 0, j = 3; i < 11; ++i) {
            tform.val[i] = *rotmat++;
            if ((i + 1) % j == 0) {
                j += 4;
                i++;
            }
        }
        tform.val[3] = position[0];
        tform.val[7] = position[1];
        tform.val[11] = position[2];
        tform.val[15] = 1.0f;
        return tform;
    }

    void update(const Pose& deltaPose) {
        this->position -= deltaPose.position;
        cv::Mat r1, r2;
        cv::Rodrigues(this->rodrigues, r1);
        cv::Rodrigues(deltaPose.rodrigues, r2);
        r1 = r1 * r2.t();
        cv::Rodrigues(r1, this->rodrigues);
    }

    std::string toString() {
        std::ostringstream stringStream;
        stringStream << "position = " << position << " rotVec = " << rodrigues << std::endl;
        return stringStream.str();
    }

    Pose& operator=(const Pose& rhs) {
        // Only do assignment if RHS is a different object from this.
        if (this != &rhs) {
            position = rhs.position;
            rodrigues = rhs.rodrigues;
        }
        return *this;
    }
private:
    // -------------------------
    // Disabling default copy constructor and default
    // assignment operator.
    // -------------------------
    //Pose(const Pose& ref);
    //Pose& operator=(const Pose& ref);
    cv::Vec3f rodrigues;
    cv::Vec3f position;
};
#endif /* __cplusplus */
#endif /* POSE_HPP */

