/*
By downloading, copying, installing or using the software you agree to this
license. If you do not agree to this license, do not download, install,
copy or use the software.
                          License Agreement
               For Open Source Computer Vision Library
                       (3-clause BSD License)
Copyright (C) 2013, OpenCV Foundation, all rights reserved.
Third party copyrights are property of their respective owners.
Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:
  * Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.
  * Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.
  * Neither the names of the copyright holders nor the names of the contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.
This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are
disclaimed. In no event shall copyright holders or contributors be liable for
any direct, indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
*/

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv_odometry/rgbd.hpp>
//#include <opencv2/core/private.hpp>
#include <iostream>
#include <list>
#include <set>
#include <limits>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/opencv.hpp>


#include <sophus/se3.hpp>
#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>
#include <Eigen/Geometry> 
#include <opencv2/features2d.hpp>

#include <chrono>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include<math.h>   

#define UNKNOWN_FLOW_THRESH 1e9  
#if defined(HAVE_EIGEN) && EIGEN_WORLD_VERSION == 3
#define HAVE_EIGEN3_HERE
#include <Eigen/Core>
#include <unsupported/Eigen/MatrixFunctions>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#endif

using namespace std;

namespace cv
{
namespace rgbd
{
using namespace cv;
enum
{
    RGBD_ODOMETRY = 1,
    ICP_ODOMETRY = 2,
    MERGED_ODOMETRY = RGBD_ODOMETRY + ICP_ODOMETRY
};

const int sobelSize = 3;
const double sobelScale = 1./8.;
int normalWinSize = 5;
//int normalMethod = RgbdNormals::RGBD_NORMALS_METHOD_FALS;
int normalMethod = RgbdNormals::RGBD_NORMALS_METHOD_LIYANG;
typedef vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;
typedef vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> VecVector3d;

double Average(vector<double> v)
{      double sum=0;
       for(int i=0;i<v.size();i++)
               sum+=v[i];
       return sum/v.size();
       // check if average converges
}
//DEVIATION
double Deviation(vector<double> v, double ave)
{
       double E=0;
       for(int i=0;i<v.size();i++){
               E+=(v[i] - ave)*(v[i] - ave);
       }
       return sqrt(E/v.size());
}
void find_feature_matches(const Mat &img_1, const Mat &img_2,
                           std::vector<KeyPoint> &keypoints_1,
			                  std::vector<KeyPoint> &keypoints_2,
                           std::vector<DMatch> &matches) {
   Mat descriptors_1, descriptors_2, descriptors_3, descriptors_4;
   Ptr<FeatureDetector> detector = ORB::create();
   Ptr<DescriptorExtractor> descriptor = ORB::create();
   Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
   detector->detect(img_1, keypoints_1);
   detector->detect(img_2, keypoints_2);
//    cout << "find1" << endl;
//    cout << "keypoints_1.size() "<< keypoints_1.size() << " keypoints_2.size() " << keypoints_2.size() << endl;
    // descriptor->compute(img_1, keypoints_1, descriptors_1);
    // descriptor->compute(img_2, keypoints_2, descriptors_2);
 }
 void find_feature_matches_another(const Mat &img_1, const Mat &img_2,
                           std::vector<KeyPoint> &keypoints_1,
			                  std::vector<KeyPoint> &keypoints_2,
                           std::vector<DMatch> &matches
                           ) {
   Mat descriptors_1, descriptors_2, descriptors_3, descriptors_4;
   Ptr<FeatureDetector> detector = AgastFeatureDetector::create();
   Ptr<DescriptorExtractor> descriptor = AgastFeatureDetector::create();
   Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
   detector->detect(img_1, keypoints_1);
   detector->detect(img_2, keypoints_2);
//    cout << "find2" << endl;
//    cout << "keypoints_1.size() "<< keypoints_1.size() << " keypoints_2.size() " << keypoints_2.size() << endl;
   // if(keypoints_1.size() != 0 && keypoints_2.size() != 0){
      for (int i=0; i< keypoints_1.size(); i++){
         // cout << "keypoint1 " << keypoints_1[i].pt << "keypoint2 " << keypoints_2[i].pt << endl;
      }
    //   descriptor->compute(img_1, keypoints_1, descriptors_1);
    //   descriptor->compute(img_2, keypoints_2, descriptors_2);
   // }
   int eee = descriptors_1.empty();
   int ddd = descriptors_2.empty();
 }
Point2d pixel2cam(const Point2d &p, const Mat &K) {
   return Point2d
     (
       (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
       (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
     );
 }
double calc_residual(
   const VecVector3d &points_3d,
  const VecVector2d &points_2d,
  Sophus::SE3d &pose,
  const Mat &K,
  vector<double>& residuals,
  const Mat& resultRt,
  int iter
){
    cout << K << endl;
    cout << "resi " << resultRt << endl;
   double fx = K.at<double>(0, 0);
  double fy = K.at<double>(1, 1);
  double cx = K.at<double>(0, 2);
  double cy = K.at<double>(1, 2);
  vector<double> res_std;
  Eigen::Matrix<double, 4, 4> Rt;
    cv2eigen(resultRt, Rt);
cout << "resi2 " << Rt << endl;
cout << points_3d.size() << endl;
  for (int i=0; i<points_3d.size(); i++){
    //   cout << points_3d[i].homogeneous().transpose() << "    " << (Rt * (points_3d[i].homogeneous())).transpose() << "    " << (Rt * (points_3d[i].homogeneous())).transpose().hnormalized() << endl;
    Eigen::Vector3d pc = (Rt * (points_3d[i].homogeneous())).transpose().hnormalized();
    
    //  Eigen::Vector3d pc = pose * points_3d[i];
     Eigen::Vector2d proj(fx * pc[0] / pc[2] + cx, fy * pc[1] / pc[2] + cy);
     
     Eigen::Vector2d error = points_2d[i] - proj;
     
     residuals.push_back(error.squaredNorm());
      if (isnan(pc[2]) == false) {
         res_std.push_back(error.squaredNorm());
      }
  }
  double total = 0;
  cout << residuals.size() << endl;
  
  for(int i=0; i<residuals.size(); i++){
      total += residuals[i] * residuals[i];
  }
    cout << "total residual square " << total << endl;
  double avg = Average(res_std); 
  double std = Deviation(res_std,avg);
  cout << "average " << avg << endl;
  return std;
}

void calculatePnPInliers(const Mat& frame0,
                         const Mat& frame1,
                         const Mat& depthF0,
                         const Mat& depthF1,
                         vector<double>& weight,
                        Sophus::SE3d& pose_gn,
                        VecVector3d& pts_3d_eigen,
                        VecVector2d& pts_2d_eigen,
                        const Mat& resultRt,
                        const Mat& K,
                        int iter
                         ){
    Mat image, depth, image1, depth1, depth_flt, depth_flt1;
    cvtColor(frame0, image1, COLOR_GRAY2BGR);
    depth_flt1 = depthF0;
    cvtColor(frame1, image, COLOR_GRAY2BGR);
    
    // cout << frame0 << endl;
    depth_flt = depthF1;
    // cout << CV_16UC1 << " " << image.type() << " " << image1.type() << " " << depth.type() << " " << depth1.type() << endl;
    // cout << depth_flt1 << endl;
    depth1 = depth_flt1;
    depth = depth_flt;
    // exit(1);
    // depth_flt.convertTo(depth, CV_16UC1, 1.f/1.f);
    // depth_flt1.convertTo(depth1, CV_16UC1, 1.f/1.f);
    // std::swap(depth1, depth);
    // depth = depth_flt;
    // depth1 = depth_flt1;
    // cout << CV_16UC1 << " " << image.type() << " " << image1.type() << " " << depth.type() << " " << depth1.type() << endl;
    // cout << "hi " << image.channels() << " " << image1.channels() << endl;
    // CV_Assert(!image.empty());
    CV_Assert(!depth.empty());
    // CV_Assert(depth.type() == CV_16UC1);
    // CV_Assert(!image1.empty());
    CV_Assert(!depth1.empty());
    // CV_Assert(depth1.type() == CV_16UC1);
    Mat prevgray, gray, flow, cflow;  
    // cvtColor(frame0, prevgray, COLOR_BGR2GRAY);
    // cvtColor(frame1, gray, COLOR_BGR2GRAY);
    // cout << CV_16UC1 << " " << prevgray.type() << " " << gray.type() << " " << depth.type() << " " << depth1.type() << endl;
    // cout << "hi " << prevgray.channels() << " " << gray.channels() << endl;
    // std::swap(image1, image);  
    
    std::vector<KeyPoint> keypoints_1, keypoints_2, key1, key2;
    vector<DMatch> matches;
    Ptr<FeatureDetector> detector = ORB::create();
    // detector->detect(image1, key1);
    // detector->detect(image, key2);
    detector->detect(frame0, key1);
    detector->detect(frame1, key2);

    if(key1.size() == 0 || key2.size() == 0){
        // find_feature_matches_another(image1, image, keypoints_1, keypoints_2, matches);
        find_feature_matches_another(frame0, frame1, keypoints_1, keypoints_2, matches);
        cout << "第二個: " <<  "一共找到了" << matches.size() << "组匹配点" << endl;
    }
    else{
        // find_feature_matches(image1, image, keypoints_1, keypoints_2, matches);
        find_feature_matches(frame0, frame1, keypoints_1, keypoints_2, matches);
        cout << "第一個: " <<"一共找到了" << matches.size() << "组匹配点" << endl;
    }
    // exit(1);
    vector<Point2f> pt1, pt2;
    for (auto &kp: keypoints_1) pt1.push_back(kp.pt);
    vector<uchar> status;
    vector<float> error;
    cv::calcOpticalFlowPyrLK(frame0, frame1, pt1, pt2, status, error);
    // cv::calcOpticalFlowPyrLK(image1, image, pt1, pt2, status, error);
    vector<Point2f> right_points_to_find;
    vector<int> right_points_to_find_back_index;
    for (unsigned int i=0; i<status.size(); i++) {
        if (status[i] && error[i] < 20.0) {
            right_points_to_find_back_index.push_back(i);
            right_points_to_find.push_back(pt2[i]);
        } else {
            status[i] = 0; // a bad flow
        }
    }
    cout << right_points_to_find.size() << endl;
    Mat right_points_to_find_flat = Mat(right_points_to_find).reshape(1,right_points_to_find.size()); //flatten array
    vector<Point2f> right_features; // detected features
    std::vector<cv::KeyPoint>::iterator it;
    for( it= keypoints_2.begin(); it!= keypoints_2.end();it++){
        right_features.push_back(it->pt);
    }
    Mat right_features_flat = Mat(right_features).reshape(1,right_features.size());
    BFMatcher matcher(NORM_L2);
    vector<vector<DMatch>> nearest_neighbors;
    matcher.radiusMatch(right_points_to_find_flat, right_features_flat, nearest_neighbors, 2.0f);
    std::set<int> found_in_right_points; // for duplicate prevention
    for(int i=0; i<nearest_neighbors.size(); i++) {
        DMatch _m;
        if(nearest_neighbors[i].size() == 1) {
        _m = nearest_neighbors[i][0]; // only one neighbor
        } else if(nearest_neighbors[i].size() > 1) {
            double ratio = nearest_neighbors[i][0].distance / nearest_neighbors[i][1].distance;
            if(ratio < 0.5) { // not too close //0.38
                _m = nearest_neighbors[i][0];
            }else { // too close – we cannot tell which is better
                continue; // did not pass ratio test – throw away
            }
        } else {
            continue; // no neighbors
        }
        if (found_in_right_points.find(_m.trainIdx) == found_in_right_points.end()) {
            _m.queryIdx = right_points_to_find_back_index[_m.queryIdx];
            matches.push_back(_m); // add this match
            found_in_right_points.insert(_m.trainIdx);
        }
    }
    cout<< "pruned " << matches.size() << " / " << nearest_neighbors.size() << " matches" << endl;
    // Mat d1 = imread(depth1, IMREAD_UNCHANGED);       // 深度图为16位无符号数，单通道图像
    // Mat K = (Mat_<double>(3, 3) << 517.3f, 0, 318.6f, 0, 516.5f, 255.3f, 0, 0, 1);
    vector<Point3f> pts_3d;
    vector<Point2f> pts_2d;
    vector<Point3f> pts_3d_nxt;
    int index = 1;
    std::vector<KeyPoint> keys1, keys2;
    std::vector<cv::Point2f> points1, points2;
    for (DMatch m:matches) {
        float d = depth1.ptr<float>(int(keypoints_1[m.queryIdx].pt.y))[int(keypoints_1[m.queryIdx].pt.x)];
        float d_nxt = depth.ptr<float>(int(keypoints_2[m.trainIdx].pt.y))[int(keypoints_2[m.trainIdx].pt.x)];
        // ushort d = depth1.at<double>(int(keypoints_1[m.queryIdx].pt.x),int(keypoints_1[m.queryIdx].pt.y));
        // exit(1);
        // cout << "original depth " << d << " " << depth1.at<float>(int(keypoints_1[m.queryIdx].pt.y),int(keypoints_1[m.queryIdx].pt.x)) << " " << int(keypoints_1[m.queryIdx].pt.y) << " " << int(keypoints_1[m.queryIdx].pt.x) << endl;
        if (d == 0 || d_nxt == 0 || isnan(d) || isnan(d_nxt)){
            continue;
        }
        float dd = d / 1.0;
        // float dd_nxt = d_nxt / 1.0;
        // cout << "orig d" << d << " " << dd << endl;
        Point2d p1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
        keys1.push_back(keypoints_1[m.queryIdx]);
        keys2.push_back(keypoints_2[m.trainIdx]);
        points1.push_back(keypoints_1[m.queryIdx].pt);
        points2.push_back(keypoints_2[m.trainIdx].pt);
        pts_3d.push_back(Point3f(p1.x * dd, p1.y * dd, dd));
        // pts_3d_nxt.push_back(Point3f(p2.x * dd_nxt, p2.y * dd_nxt, dd_nxt));
        pts_2d.push_back(keypoints_2[m.trainIdx].pt);
        cout << dd << " keypoints_1[m.queryIdx].pt " << keypoints_1[m.queryIdx].pt << " keypoints_2[m.trainIdx].pt " << keypoints_2[m.trainIdx].pt << endl;        
        index += 1;
    }
    cout << "3d-2d pairs: " << pts_3d.size() << " " << pts_2d.size() << endl;
    // exit(1);
    for (size_t i = 0; i < pts_3d.size(); ++i) {
        pts_3d_eigen.push_back(Eigen::Vector3d(pts_3d[i].x, pts_3d[i].y, pts_3d[i].z));
        pts_2d_eigen.push_back(Eigen::Vector2d(pts_2d[i].x, pts_2d[i].y));
    }
    bool b = false;
    keypoints_1.clear();
    keypoints_1 = keys1;
    keypoints_2.clear();
    keypoints_2 = keys2;
    Mat r, t, inliers;
    cout << "3d-2d pairs: " << pts_3d.size() << " " << pts_2d.size() << endl;
    solvePnPRansac(pts_3d, pts_2d, K, Mat(), r, t, b, 1000, 6.0, 0.99, inliers, SOLVEPNP_ITERATIVE);
    Mat R;
    cv::Rodrigues(r, R);
    Mat output;
    Mat Rt_ba;
    // VecVector3d pts_3d_eigen;
    // VecVector2d pts_2d_eigen;
    cout << "3d-2d pairs: " << pts_3d.size() << " " << pts_2d.size() << endl;
    // for (size_t i = 0; i < pts_3d.size(); ++i) {
    //     pts_3d_eigen.push_back(Eigen::Vector3d(pts_3d[i].x, pts_3d[i].y, pts_3d[i].z));
    //     pts_2d_eigen.push_back(Eigen::Vector2d(pts_2d[i].x, pts_2d[i].y));
    // }
    int mode = 0; // 0=huber
    vector<double> residuals;

    double res_std = calc_residual(pts_3d_eigen, pts_2d_eigen, pose_gn, K, residuals, resultRt, iter);
    cout << "deviation: " << res_std << endl;
    double huber_k = 1.345 * res_std;
    // vector<double> weight;
    if(mode == 0){
        for (int j=0; j<residuals.size(); j++){
            if(residuals[j] <= huber_k){
                weight.push_back(1.0);
                // cout << "weight " << 1.0 << endl;
            }else {
                weight.push_back(huber_k/residuals[j]);
                // cout << "weight " << huber_k/residuals[j] << endl;
            }

        }
    }else {
        for (int j=0; j<residuals.size(); j++){
            weight.push_back(0.0);
            // cout << "weight " << 0.0 << endl;
        }
    }
    
}

static inline
void setDefaultIterCounts(Mat& iterCounts)
{
    // iterCounts = Mat(Vec4i(7,7,7,10));
    iterCounts = Mat(Vec<int,1>(10));
}

static inline
void setDefaultMinGradientMagnitudes(Mat& minGradientMagnitudes)
{
    // minGradientMagnitudes = Mat(Vec4f(10,10,10,10));
    minGradientMagnitudes = Mat(Vec<float,1>(10));
}

static
void buildPyramidCameraMatrix(const Mat& cameraMatrix, int levels, std::vector<Mat>& pyramidCameraMatrix) //levels = 4
{
    // cout << cameraMatrix << " " << levels << " " << pyramidCameraMatrix.size() << endl;
    pyramidCameraMatrix.resize(levels);

    Mat cameraMatrix_dbl;
    cameraMatrix.convertTo(cameraMatrix_dbl, CV_64FC1);
    // cout << "cameraMatrix_dbl" << cameraMatrix_dbl << endl;
    for(int i = 0; i < levels; i++)
    {
        // cout << "index: " << i << endl;
        Mat levelCameraMatrix = i == 0 ? cameraMatrix_dbl : 0.5f * pyramidCameraMatrix[i-1];
        levelCameraMatrix.at<double>(2,2) = 1.;
        pyramidCameraMatrix[i] = levelCameraMatrix;
        // cout << i << " levelCameraMatrix2" << levelCameraMatrix << endl;

    }
    // exit(1);
}

static inline
void checkImage(const Mat& image)
{
    if(image.empty())
        CV_Error(Error::StsBadSize, "Image is empty.");
    if(image.type() != CV_8UC1)
        CV_Error(Error::StsBadSize, "Image type has to be CV_8UC1.");
}

static inline
void checkDepth(const Mat& depth, const Size& imageSize)
{
    if(depth.empty())
        CV_Error(Error::StsBadSize, "Depth is empty.");
    if(depth.size() != imageSize)
        CV_Error(Error::StsBadSize, "Depth has to have the size equal to the image size.");
    if(depth.type() != CV_32FC1)
        CV_Error(Error::StsBadSize, "Depth type has to be CV_32FC1.");
}

static inline
void checkMask(const Mat& mask, const Size& imageSize)
{
    if(!mask.empty())
    {
        if(mask.size() != imageSize)
            CV_Error(Error::StsBadSize, "Mask has to have the size equal to the image size.");
        if(mask.type() != CV_8UC1)
            CV_Error(Error::StsBadSize, "Mask type has to be CV_8UC1.");
    }
}

static inline
void checkNormals(const Mat& normals, const Size& depthSize)
{
    if(normals.size() != depthSize)
        CV_Error(Error::StsBadSize, "Normals has to have the size equal to the depth size.");
    if(normals.type() != CV_32FC3)
        CV_Error(Error::StsBadSize, "Normals type has to be CV_32FC3.");
}

static
void preparePyramidImage(const Mat& image, std::vector<Mat>& pyramidImage, size_t levelCount)
{
    if(!pyramidImage.empty())
    {
        if(pyramidImage.size() < levelCount)
            CV_Error(Error::StsBadSize, "Levels count of pyramidImage has to be equal or less than size of iterCounts.");

        CV_Assert(pyramidImage[0].size() == image.size());
        for(size_t i = 0; i < pyramidImage.size(); i++)
            CV_Assert(pyramidImage[i].type() == image.type());
    }
    else
        buildPyramid(image, pyramidImage, (int)levelCount - 1);
}

static
void preparePyramidDepth(const Mat& depth, std::vector<Mat>& pyramidDepth, size_t levelCount)
{
    if(!pyramidDepth.empty())
    {
        if(pyramidDepth.size() < levelCount)
            CV_Error(Error::StsBadSize, "Levels count of pyramidDepth has to be equal or less than size of iterCounts.");

        CV_Assert(pyramidDepth[0].size() == depth.size());
        for(size_t i = 0; i < pyramidDepth.size(); i++)
            CV_Assert(pyramidDepth[i].type() == depth.type());
    }
    else
        buildPyramid(depth, pyramidDepth, (int)levelCount - 1);
}

static
void preparePyramidMask(const Mat& mask, const std::vector<Mat>& pyramidDepth, float minDepth, float maxDepth,
                        const std::vector<Mat>& pyramidNormal,
                        std::vector<Mat>& pyramidMask)
{
    minDepth = std::max(0.f, minDepth);

    if(!pyramidMask.empty())
    {
        if(pyramidMask.size() != pyramidDepth.size())
            CV_Error(Error::StsBadSize, "Levels count of pyramidMask has to be equal to size of pyramidDepth.");

        for(size_t i = 0; i < pyramidMask.size(); i++)
        {
            CV_Assert(pyramidMask[i].size() == pyramidDepth[i].size());
            CV_Assert(pyramidMask[i].type() == CV_8UC1);
        }
    }
    else
    {
        Mat validMask;
        if(mask.empty())
            validMask = Mat(pyramidDepth[0].size(), CV_8UC1, Scalar(255));
        else
            validMask = mask.clone();

        //cout << "liyang test" << validMask << endl;
        //exit(1);
        buildPyramid(validMask, pyramidMask, (int)pyramidDepth.size() - 1);
        // cout << pyramidMask.size() << " " << mask.size() << " " << mask.empty() << " " << pyramidMask.empty() << endl;
        for(size_t i = 0; i < pyramidMask.size(); i++)
        {
            Mat levelDepth = pyramidDepth[i].clone();
            // cout << "checksize " << pyramidDepth[i].size() << " " << pyramidMask[i].size() << endl;
            patchNaNs(levelDepth, 0);
            // cout << minDepth << " " << maxDepth << endl;
            Mat& levelMask = pyramidMask[i];
            levelMask &= (levelDepth > minDepth) & (levelDepth < maxDepth);

            if(!pyramidNormal.empty())
            {
                CV_Assert(pyramidNormal[i].type() == CV_32FC3);
                CV_Assert(pyramidNormal[i].size() == pyramidDepth[i].size());
                Mat levelNormal = pyramidNormal[i].clone();
                
                Mat validNormalMask = levelNormal == levelNormal; // otherwise it's Nan
                // cout << "normalsize " << pyramidNormal[i].size() << validNormalMask.size() << endl; 
                CV_Assert(validNormalMask.type() == CV_8UC3);

                std::vector<Mat> channelMasks;
                split(validNormalMask, channelMasks);
                validNormalMask = channelMasks[0] & channelMasks[1] & channelMasks[2];

                levelMask &= validNormalMask;
            }
        }
    }
}

static
void preparePyramidCloud(const std::vector<Mat>& pyramidDepth, const Mat& cameraMatrix, std::vector<Mat>& pyramidCloud)
{
    if(!pyramidCloud.empty())
    {
        if(pyramidCloud.size() != pyramidDepth.size())
            CV_Error(Error::StsBadSize, "Incorrect size of pyramidCloud.");

        for(size_t i = 0; i < pyramidDepth.size(); i++)
        {
            CV_Assert(pyramidCloud[i].size() == pyramidDepth[i].size());
            CV_Assert(pyramidCloud[i].type() == CV_32FC3);
        }
    }
    else
    {
        std::vector<Mat> pyramidCameraMatrix;
        buildPyramidCameraMatrix(cameraMatrix, (int)pyramidDepth.size(), pyramidCameraMatrix);

        pyramidCloud.resize(pyramidDepth.size());
        for(size_t i = 0; i < pyramidDepth.size(); i++)
        {
            Mat cloud;
            depthTo3d(pyramidDepth[i], pyramidCameraMatrix[i], cloud);
            pyramidCloud[i] = cloud;
        }
    }
}

static
void preparePyramidSobel(const std::vector<Mat>& pyramidImage, int dx, int dy, std::vector<Mat>& pyramidSobel)
{
    if(!pyramidSobel.empty())
    {
        if(pyramidSobel.size() != pyramidImage.size())
            CV_Error(Error::StsBadSize, "Incorrect size of pyramidSobel.");

        for(size_t i = 0; i < pyramidSobel.size(); i++)
        {
            CV_Assert(pyramidSobel[i].size() == pyramidImage[i].size());
            CV_Assert(pyramidSobel[i].type() == CV_16SC1);
        }
    }
    else
    {
        pyramidSobel.resize(pyramidImage.size());
        
        for(size_t i = 0; i < pyramidImage.size(); i++)
        {
            Sobel(pyramidImage[i], pyramidSobel[i], CV_16S, dx, dy, sobelSize);
            // cout << "sobelimage " << pyramidImage.size() << " " << pyramidSobel[i].size() << endl;
        }
        // cout << pyramidSobel.size() << " ========" << endl;
    }
}

static
void randomSubsetOfMask(Mat& mask, float part)
{
    const int minPointsCount = 1000; // minimum point count (we can process them fast)
    const int nonzeros = countNonZero(mask);
    const int needCount = std::max(minPointsCount, int(mask.total() * part));
    if(needCount < nonzeros)
    {
        RNG rng;
        Mat subset(mask.size(), CV_8UC1, Scalar(0));

        int subsetSize = 0;
        while(subsetSize < needCount)
        {
            int y = rng(mask.rows);
            int x = rng(mask.cols);
            if(mask.at<uchar>(y,x))
            {
                subset.at<uchar>(y,x) = 255;
                mask.at<uchar>(y,x) = 0;
                subsetSize++;
            }
        }
        mask = subset;
    }
}

static
void preparePyramidTexturedMask(const std::vector<Mat>& pyramid_dI_dx, const std::vector<Mat>& pyramid_dI_dy,
                                const std::vector<float>& minGradMagnitudes, const std::vector<Mat>& pyramidMask, double maxPointsPart,
                                std::vector<Mat>& pyramidTexturedMask)
{
    if(!pyramidTexturedMask.empty())
    {
        if(pyramidTexturedMask.size() != pyramid_dI_dx.size())
            CV_Error(Error::StsBadSize, "Incorrect size of pyramidTexturedMask.");

        for(size_t i = 0; i < pyramidTexturedMask.size(); i++)
        {
            CV_Assert(pyramidTexturedMask[i].size() == pyramid_dI_dx[i].size());
            CV_Assert(pyramidTexturedMask[i].type() == CV_8UC1);
        }
    }
    else
    {
        const float sobelScale2_inv = 1.f / (float)(sobelScale * sobelScale); //sobelscale=1/8
        pyramidTexturedMask.resize(pyramid_dI_dx.size());
        for(size_t i = 0; i < pyramidTexturedMask.size(); i++)
        {
            const float minScaledGradMagnitude2 = minGradMagnitudes[i] * minGradMagnitudes[i] * sobelScale2_inv;
            const Mat& dIdx = pyramid_dI_dx[i];
            const Mat& dIdy = pyramid_dI_dy[i];

            Mat texturedMask(dIdx.size(), CV_8UC1, Scalar(0));

            for(int y = 0; y < dIdx.rows; y++)
            {
                const short *dIdx_row = dIdx.ptr<short>(y);
                const short *dIdy_row = dIdy.ptr<short>(y);
                uchar *texturedMask_row = texturedMask.ptr<uchar>(y);
                for(int x = 0; x < dIdx.cols; x++)
                {
                    float magnitude2 = static_cast<float>(dIdx_row[x] * dIdx_row[x] + dIdy_row[x] * dIdy_row[x]);
                    if(magnitude2 >= minScaledGradMagnitude2)
                        texturedMask_row[x] = 255;
                }
            }
            pyramidTexturedMask[i] = texturedMask & pyramidMask[i];

            randomSubsetOfMask(pyramidTexturedMask[i], (float)maxPointsPart);
        }
    }
}

static
void preparePyramidNormals(const Mat& normals, const std::vector<Mat>& pyramidDepth, std::vector<Mat>& pyramidNormals)
{
    if(!pyramidNormals.empty())
    {
        if(pyramidNormals.size() != pyramidDepth.size())
            CV_Error(Error::StsBadSize, "Incorrect size of pyramidNormals.");

        for(size_t i = 0; i < pyramidNormals.size(); i++)
        {
            CV_Assert(pyramidNormals[i].size() == pyramidDepth[i].size());
            CV_Assert(pyramidNormals[i].type() == CV_32FC3);
        }
    }
    else
    {
        buildPyramid(normals, pyramidNormals, (int)pyramidDepth.size() - 1);
        // renormalize normals
        for(size_t i = 1; i < pyramidNormals.size(); i++)
        {
            Mat& currNormals = pyramidNormals[i];
            for(int y = 0; y < currNormals.rows; y++)
            {
                Point3f* normals_row = currNormals.ptr<Point3f>(y);
                for(int x = 0; x < currNormals.cols; x++)
                {
                    double nrm = norm(normals_row[x]);
                    normals_row[x] *= 1./nrm;
                }
            }
        }
    }
}

static
void preparePyramidNormalsMask(const std::vector<Mat>& pyramidNormals, const std::vector<Mat>& pyramidMask, double maxPointsPart,
                               std::vector<Mat>& pyramidNormalsMask)
{
    if(!pyramidNormalsMask.empty())
    {
        if(pyramidNormalsMask.size() != pyramidMask.size())
            CV_Error(Error::StsBadSize, "Incorrect size of pyramidNormalsMask.");

        for(size_t i = 0; i < pyramidNormalsMask.size(); i++)
        {
            CV_Assert(pyramidNormalsMask[i].size() == pyramidMask[i].size());
            CV_Assert(pyramidNormalsMask[i].type() == pyramidMask[i].type());
        }
    }
    else
    {
        pyramidNormalsMask.resize(pyramidMask.size());

        for(size_t i = 0; i < pyramidNormalsMask.size(); i++)
        {
            pyramidNormalsMask[i] = pyramidMask[i].clone();
            Mat& normalsMask = pyramidNormalsMask[i];
            for(int y = 0; y < normalsMask.rows; y++)
            {
                const Vec3f *normals_row = pyramidNormals[i].ptr<Vec3f>(y);
                uchar *normalsMask_row = pyramidNormalsMask[i].ptr<uchar>(y);
                for(int x = 0; x < normalsMask.cols; x++)
                {
                    Vec3f n = normals_row[x];
                    if(cvIsNaN(n[0]))
                    {
                        CV_DbgAssert(cvIsNaN(n[1]) && cvIsNaN(n[2]));
                        normalsMask_row[x] = 0;
                    }
                }
            }
            randomSubsetOfMask(normalsMask, (float)maxPointsPart);
        }
    }
}

///////////////////////////////////////////////////////////////////////////////////////
static
void computeProjectiveMatrixInv(const Mat& ksi, Mat& Rt)
{
    CV_Assert(ksi.size() == Size(1,6) && ksi.type() == CV_64FC1);

// #ifdef HAVE_EIGEN3_HERE
//     const double* ksi_ptr = ksi.ptr<const double>();
//     Eigen::Matrix<double,4,4> twist, g;
//     twist << 0.,          -ksi_ptr[2], ksi_ptr[1],  ksi_ptr[3],
//              ksi_ptr[2],  0.,          -ksi_ptr[0], ksi_ptr[4],
//              -ksi_ptr[1], ksi_ptr[0],  0,           ksi_ptr[5],
//              0.,          0.,          0.,          0.;
//     g = twist.exp();

//     eigen2cv(g, Rt);
// #else
    // TODO: check computeProjectiveMatrix when there is not eigen library,
    //       because it gives less accurate pose of the camera
    Rt = Mat::eye(4, 4, CV_64FC1);
    Mat R = Rt(Rect(0,0,3,3)); // 左上角x 左上角y width height
    // cout << "Rt1" << Rt << R << endl;
    // cout << "compute" << ksi.rowRange(0,3) << ksi.rowRange(0,2) << ksi << endl;
    Mat rvec = ksi.rowRange(3,6);

    Rodrigues(rvec, R);
    // cout << "Rt2" << Rt << R << endl;
    Rt.at<double>(0,3) = ksi.at<double>(0);
    Rt.at<double>(1,3) = ksi.at<double>(1);
    Rt.at<double>(2,3) = ksi.at<double>(2);
    // cout << "Rt3" << Rt << endl;
// #endif
}

static
void computeProjectiveMatrix(const Mat& ksi, Mat& Rt)
{
    CV_Assert(ksi.size() == Size(1,6) && ksi.type() == CV_64FC1);

#ifdef HAVE_EIGEN3_HERE
    const double* ksi_ptr = ksi.ptr<const double>();
    Eigen::Matrix<double,4,4> twist, g;
    twist << 0.,          -ksi_ptr[2], ksi_ptr[1],  ksi_ptr[3],
             ksi_ptr[2],  0.,          -ksi_ptr[0], ksi_ptr[4],
             -ksi_ptr[1], ksi_ptr[0],  0,           ksi_ptr[5],
             0.,          0.,          0.,          0.;
    g = twist.exp();

    eigen2cv(g, Rt);
#else
    // TODO: check computeProjectiveMatrix when there is not eigen library,
    //       because it gives less accurate pose of the camera
    Rt = Mat::eye(4, 4, CV_64FC1);

    Mat R = Rt(Rect(0,0,3,3)); // 左上角x 左上角y width height
    cout << "Rt1" << Rt << R << endl;
    // cout << "compute" << ksi.rowRange(0,3) << ksi.rowRange(0,2) << ksi << endl;
    Mat rvec = ksi.rowRange(0,3);

    Rodrigues(rvec, R);
    // cout << "Rt2" << Rt << R << endl;
    Rt.at<double>(0,3) = ksi.at<double>(3);
    Rt.at<double>(1,3) = ksi.at<double>(4);
    Rt.at<double>(2,3) = ksi.at<double>(5);
    // cout << "Rt3" << Rt << endl;
#endif
}

static
void computeCorresps(const Mat& K, const Mat& K_inv, const Mat& Rt,
                     const Mat& depth0, const Mat& validMask0,
                     const Mat& depth1, const Mat& selectMask1, float maxDepthDiff,
                     Mat& _corresps)
{
    CV_Assert(K.type() == CV_64FC1);
    CV_Assert(K_inv.type() == CV_64FC1);
    CV_Assert(Rt.type() == CV_64FC1);

    Mat corresps(depth1.size(), CV_16SC2, Scalar::all(-1));

    Rect r(0, 0, depth1.cols, depth1.rows);
    Mat Kt = Rt(Rect(3,0,1,3)).clone();
    Kt = K * Kt;
    const double * Kt_ptr = Kt.ptr<const double>();

    AutoBuffer<float> buf(3 * (depth1.cols + depth1.rows));
    float *KRK_inv0_u1 = buf;
    float *KRK_inv1_v1_plus_KRK_inv2 = KRK_inv0_u1 + depth1.cols;
    float *KRK_inv3_u1 = KRK_inv1_v1_plus_KRK_inv2 + depth1.rows;
    float *KRK_inv4_v1_plus_KRK_inv5 = KRK_inv3_u1 + depth1.cols;
    float *KRK_inv6_u1 = KRK_inv4_v1_plus_KRK_inv5 + depth1.rows;
    float *KRK_inv7_v1_plus_KRK_inv8 = KRK_inv6_u1 + depth1.cols;
    {
        Mat R = Rt(Rect(0,0,3,3)).clone();

        Mat KRK_inv = K * R * K_inv;
        const double * KRK_inv_ptr = KRK_inv.ptr<const double>();
        for(int u1 = 0; u1 < depth1.cols; u1++)
        {
            KRK_inv0_u1[u1] = (float)(KRK_inv_ptr[0] * u1);
            KRK_inv3_u1[u1] = (float)(KRK_inv_ptr[3] * u1);
            KRK_inv6_u1[u1] = (float)(KRK_inv_ptr[6] * u1);
        }

        for(int v1 = 0; v1 < depth1.rows; v1++)
        {
            KRK_inv1_v1_plus_KRK_inv2[v1] = (float)(KRK_inv_ptr[1] * v1 + KRK_inv_ptr[2]);
            KRK_inv4_v1_plus_KRK_inv5[v1] = (float)(KRK_inv_ptr[4] * v1 + KRK_inv_ptr[5]);
            KRK_inv7_v1_plus_KRK_inv8[v1] = (float)(KRK_inv_ptr[7] * v1 + KRK_inv_ptr[8]);
        }
    }

    int correspCount = 0;
    for(int v1 = 0; v1 < depth1.rows; v1++)
    {
        const float *depth1_row = depth1.ptr<float>(v1);
        const uchar *mask1_row = selectMask1.ptr<uchar>(v1);
        for(int u1 = 0; u1 < depth1.cols; u1++)
        {
            float d1 = depth1_row[u1];
            if(mask1_row[u1])
            {
                CV_DbgAssert(!cvIsNaN(d1));
                float transformed_d1 = static_cast<float>(d1 * (KRK_inv6_u1[u1] + KRK_inv7_v1_plus_KRK_inv8[v1]) +
                                                          Kt_ptr[2]);
                if(transformed_d1 > 0)
                {
                    float transformed_d1_inv = 1.f / transformed_d1;
                    int u0 = cvRound(transformed_d1_inv * (d1 * (KRK_inv0_u1[u1] + KRK_inv1_v1_plus_KRK_inv2[v1]) +
                                                           Kt_ptr[0]));
                    int v0 = cvRound(transformed_d1_inv * (d1 * (KRK_inv3_u1[u1] + KRK_inv4_v1_plus_KRK_inv5[v1]) +
                                                           Kt_ptr[1]));

                    if(r.contains(Point(u0,v0)))
                    {
                        float d0 = depth0.at<float>(v0,u0);
                        if(validMask0.at<uchar>(v0, u0) && std::abs(transformed_d1 - d0) <= maxDepthDiff)
                        {
                            CV_DbgAssert(!cvIsNaN(d0));
                            Vec2s& c = corresps.at<Vec2s>(v0,u0);
                            if(c[0] != -1)
                            {
                                int exist_u1 = c[0], exist_v1 = c[1];

                                float exist_d1 = (float)(depth1.at<float>(exist_v1,exist_u1) *
                                    (KRK_inv6_u1[exist_u1] + KRK_inv7_v1_plus_KRK_inv8[exist_v1]) + Kt_ptr[2]);

                                if(transformed_d1 > exist_d1)
                                    continue;
                            }
                            else
                                correspCount++;

                            c = Vec2s((short)u1, (short)v1);
                        }
                    }
                }
            }
        }
    }

    _corresps.create(correspCount, 1, CV_32SC4);
    Vec4i * corresps_ptr = _corresps.ptr<Vec4i>();
    for(int v0 = 0, i = 0; v0 < corresps.rows; v0++)
    {
        const Vec2s* corresps_row = corresps.ptr<Vec2s>(v0);
        for(int u0 = 0; u0 < corresps.cols; u0++)
        {
            const Vec2s& c = corresps_row[u0];
            if(c[0] != -1)
                corresps_ptr[i++] = Vec4i(u0,v0,c[0],c[1]);
        }
    }
}

static inline
void calcRgbdEquationCoeffs(double* C, double dIdx, double dIdy, const Point3f& p3d, double fx, double fy)
{
    double invz  = 1. / p3d.z,
           v0 = dIdx * fx * invz,
           v1 = dIdy * fy * invz,
           v2 = -(v0 * p3d.x + v1 * p3d.y) * invz;

    C[0] = -p3d.z * v1 + p3d.y * v2;
    C[1] =  p3d.z * v0 - p3d.x * v2;
    C[2] = -p3d.y * v0 + p3d.x * v1;
    C[3] = v0;
    C[4] = v1;
    C[5] = v2;
}

static inline
void calcRgbdEquationCoeffsRotation(double* C, double dIdx, double dIdy, const Point3f& p3d, double fx, double fy)
{
    double invz  = 1. / p3d.z,
           v0 = dIdx * fx * invz,
           v1 = dIdy * fy * invz,
           v2 = -(v0 * p3d.x + v1 * p3d.y) * invz;
    C[0] = -p3d.z * v1 + p3d.y * v2;
    C[1] =  p3d.z * v0 - p3d.x * v2;
    C[2] = -p3d.y * v0 + p3d.x * v1;
}

static inline
void calcRgbdEquationCoeffsTranslation(double* C, double dIdx, double dIdy, const Point3f& p3d, double fx, double fy)
{
    double invz  = 1. / p3d.z,
           v0 = dIdx * fx * invz,
           v1 = dIdy * fy * invz,
           v2 = -(v0 * p3d.x + v1 * p3d.y) * invz;
    C[0] = v0;
    C[1] = v1;
    C[2] = v2;
}

typedef
void (*CalcRgbdEquationCoeffsPtr)(double*, double, double, const Point3f&, double, double);

static inline
void calcICPEquationCoeffs(double* C, const Point3f& p0, const Vec3f& n1)
{
    C[0] = -p0.z * n1[1] + p0.y * n1[2];
    C[1] =  p0.z * n1[0] - p0.x * n1[2];
    C[2] = -p0.y * n1[0] + p0.x * n1[1];
    C[3] = n1[0];
    C[4] = n1[1];
    C[5] = n1[2];
}

static inline
void calcICPEquationCoeffsRotation(double* C, const Point3f& p0, const Vec3f& n1)
{
    C[0] = -p0.z * n1[1] + p0.y * n1[2];
    C[1] =  p0.z * n1[0] - p0.x * n1[2];
    C[2] = -p0.y * n1[0] + p0.x * n1[1];
}

static inline
void calcICPEquationCoeffsTranslation(double* C, const Point3f& /*p0*/, const Vec3f& n1)
{
    C[0] = n1[0];
    C[1] = n1[1];
    C[2] = n1[2];
}

typedef
void (*CalcICPEquationCoeffsPtr)(double*, const Point3f&, const Vec3f&);

static
void calcRgbdLsmMatrices(const Mat& image0, const Mat& cloud0, const Mat& Rt,
               const Mat& image1, const Mat& dI_dx1, const Mat& dI_dy1,
               const Mat& corresps, double fx, double fy, double sobelScaleIn,
               Mat& AtA, Mat& AtB, CalcRgbdEquationCoeffsPtr func, int transformDim)
{
    AtA = Mat(transformDim, transformDim, CV_64FC1, Scalar(0));
    AtB = Mat(transformDim, 1, CV_64FC1, Scalar(0));
    double* AtB_ptr = AtB.ptr<double>();

    const int correspsCount = corresps.rows;

    CV_Assert(Rt.type() == CV_64FC1);
    const double * Rt_ptr = Rt.ptr<const double>();

    AutoBuffer<float> diffs(correspsCount);
    float* diffs_ptr = diffs;

    const Vec4i* corresps_ptr = corresps.ptr<Vec4i>();

    double sigma = 0;
    for(int correspIndex = 0; correspIndex < corresps.rows; correspIndex++)
    {
         const Vec4i& c = corresps_ptr[correspIndex];
         int u0 = c[0], v0 = c[1];
         int u1 = c[2], v1 = c[3];

         diffs_ptr[correspIndex] = static_cast<float>(static_cast<int>(image0.at<uchar>(v0,u0)) -
                                                      static_cast<int>(image1.at<uchar>(v1,u1)));
         //std::cout << "====================test=======================" << diffs_ptr[0] <<  std::endl;
         //std::cout << static_cast<int>(image0.at<uchar>(v0,u0)) <<  std::endl;
         //std::cout << static_cast<int>(image1.at<uchar>(v1,u1)) <<  std::endl;
	 //exit(1);
         sigma += diffs_ptr[correspIndex] * diffs_ptr[correspIndex];
    }
    sigma = std::sqrt(sigma/correspsCount);

    std::vector<double> A_buf(transformDim);
    double* A_ptr = &A_buf[0];

    for(int correspIndex = 0; correspIndex < corresps.rows; correspIndex++)
    {
         const Vec4i& c = corresps_ptr[correspIndex];
         int u0 = c[0], v0 = c[1];
         int u1 = c[2], v1 = c[3];

         double w = sigma + std::abs(diffs_ptr[correspIndex]);
         w = w > DBL_EPSILON ? 1./w : 1.;

         double w_sobelScale = w * sobelScaleIn;

         const Point3f& p0 = cloud0.at<Point3f>(v0,u0);
         Point3f tp0;
         tp0.x = (float)(p0.x * Rt_ptr[0] + p0.y * Rt_ptr[1] + p0.z * Rt_ptr[2] + Rt_ptr[3]);
         tp0.y = (float)(p0.x * Rt_ptr[4] + p0.y * Rt_ptr[5] + p0.z * Rt_ptr[6] + Rt_ptr[7]);
         tp0.z = (float)(p0.x * Rt_ptr[8] + p0.y * Rt_ptr[9] + p0.z * Rt_ptr[10] + Rt_ptr[11]);

         func(A_ptr,
              w_sobelScale * dI_dx1.at<short int>(v1,u1),
              w_sobelScale * dI_dy1.at<short int>(v1,u1),
              tp0, fx, fy);

        for(int y = 0; y < transformDim; y++)
        {
            double* AtA_ptr = AtA.ptr<double>(y);
            for(int x = y; x < transformDim; x++)
                AtA_ptr[x] += A_ptr[y] * A_ptr[x];

            AtB_ptr[y] += A_ptr[y] * w * diffs_ptr[correspIndex];
        }
    }

    for(int y = 0; y < transformDim; y++)
        for(int x = y+1; x < transformDim; x++)
            AtA.at<double>(x,y) = AtA.at<double>(y,x);
}

static
void calcICPLsmMatrices(const Mat& cloud0, const Mat& Rt,
                        const Mat& cloud1, const Mat& normals1,
                        const Mat& corresps,
                        Mat& AtA, Mat& AtB, CalcICPEquationCoeffsPtr func, int transformDim)
{
    AtA = Mat(transformDim, transformDim, CV_64FC1, Scalar(0));
    AtB = Mat(transformDim, 1, CV_64FC1, Scalar(0));
    double* AtB_ptr = AtB.ptr<double>();

    const int correspsCount = corresps.rows;

    CV_Assert(Rt.type() == CV_64FC1);
    const double * Rt_ptr = Rt.ptr<const double>();

    AutoBuffer<float> diffs(correspsCount);
    float * diffs_ptr = diffs;

    AutoBuffer<Point3f> transformedPoints0(correspsCount);
    Point3f * tps0_ptr = transformedPoints0;

    const Vec4i* corresps_ptr = corresps.ptr<Vec4i>();

    double sigma = 0;
    for(int correspIndex = 0; correspIndex < corresps.rows; correspIndex++)
    {
        const Vec4i& c = corresps_ptr[correspIndex];
        int u0 = c[0], v0 = c[1];
        int u1 = c[2], v1 = c[3];

        const Point3f& p0 = cloud0.at<Point3f>(v0,u0);
        Point3f tp0;
        tp0.x = (float)(p0.x * Rt_ptr[0] + p0.y * Rt_ptr[1] + p0.z * Rt_ptr[2] + Rt_ptr[3]);
        tp0.y = (float)(p0.x * Rt_ptr[4] + p0.y * Rt_ptr[5] + p0.z * Rt_ptr[6] + Rt_ptr[7]);
        tp0.z = (float)(p0.x * Rt_ptr[8] + p0.y * Rt_ptr[9] + p0.z * Rt_ptr[10] + Rt_ptr[11]);

        Vec3f n1 = normals1.at<Vec3f>(v1, u1);
        Point3f v = cloud1.at<Point3f>(v1,u1) - tp0;

        tps0_ptr[correspIndex] = tp0;
        diffs_ptr[correspIndex] = n1[0] * v.x + n1[1] * v.y + n1[2] * v.z;
        //std::cout << "====================test=======================" << diffs_ptr[0] <<  std::endl;
        //exit(1);
        sigma += diffs_ptr[correspIndex] * diffs_ptr[correspIndex];
    }

    sigma = std::sqrt(sigma/correspsCount);

    std::vector<double> A_buf(transformDim);
    double* A_ptr = &A_buf[0];
    for(int correspIndex = 0; correspIndex < corresps.rows; correspIndex++)
    {
        const Vec4i& c = corresps_ptr[correspIndex];
        int u1 = c[2], v1 = c[3];

        double w = sigma + std::abs(diffs_ptr[correspIndex]);
        w = w > DBL_EPSILON ? 1./w : 1.;

        func(A_ptr, tps0_ptr[correspIndex], normals1.at<Vec3f>(v1, u1) * w);

        for(int y = 0; y < transformDim; y++)
        {
            double* AtA_ptr = AtA.ptr<double>(y);
            for(int x = y; x < transformDim; x++)
                AtA_ptr[x] += A_ptr[y] * A_ptr[x];

            AtB_ptr[y] += A_ptr[y] * w * diffs_ptr[correspIndex];
        }
    }

    for(int y = 0; y < transformDim; y++)
        for(int x = y+1; x < transformDim; x++)
            AtA.at<double>(x,y) = AtA.at<double>(y,x);
}

static
bool solveSystem(const Mat& AtA, const Mat& AtB, double detThreshold, Mat& x)
{
    double det = determinant(AtA);
    cout << "solve system " << det << " " << AtA << endl;
    if(fabs (det) < detThreshold || cvIsNaN(det) || cvIsInf(det)) // fabs: double型態絕對值
        return false;

    solve(AtA, AtB, x, DECOMP_CHOLESKY);

    return true;
}

static
bool testDeltaTransformation(const Mat& deltaRt, double maxTranslation, double maxRotation)
{
    double translation = norm(deltaRt(Rect(3, 0, 1, 3))); // fourth column with 3 rows
    // cout << "trans" << deltaRt(Rect(3, 0, 1, 3)) << translation << endl;
    Mat rvec;
    Rodrigues(deltaRt(Rect(0,0,3,3)), rvec);
    // cout << "rvec33" << rvec << endl;
    double rotation = norm(rvec) * 180. / CV_PI; //弧度轉角度
    // cout << "rotarrr" << norm(rvec) << rotation << " " << maxTranslation << " " << maxRotation << endl; // maxtrans/rot: 0.15 15
    return translation <= maxTranslation && rotation <= maxRotation;
}

static
bool RGBDICPOdometryImpl(Mat& Rt, const Mat& initRt,
                         const Ptr<OdometryFrame>& srcFrame,
                         const Ptr<OdometryFrame>& dstFrame,
                         const Mat& cameraMatrix,
                         float maxDepthDiff, const std::vector<int>& iterCounts,
                         double maxTranslation, double maxRotation,
                         int method, int transfromType)
{
    cout << maxDepthDiff  << " " << maxTranslation << " " << maxRotation << " "  << method  << endl; // 0.07 0.15 15 3
    // for (int i=0; i<iterCounts.size(); i++) {
    //     cout << i << ":" << iterCounts[i] << endl;
    // }
    // cout << "RGBDICPOdometryImpl " << srcFrame->depth.type() << " " << dstFrame->depth.type() << endl;
    int transformDim = -1;
    CalcRgbdEquationCoeffsPtr rgbdEquationFuncPtr = 0;
    CalcICPEquationCoeffsPtr icpEquationFuncPtr = 0;
    switch(transfromType)
    {
    case Odometry::RIGID_BODY_MOTION:
        transformDim = 6;
        rgbdEquationFuncPtr = calcRgbdEquationCoeffs;
        icpEquationFuncPtr = calcICPEquationCoeffs;
        // cout << "rigid" << rgbdEquationFuncPtr << " " << icpEquationFuncPtr << endl; // 1 1
        break;
    case Odometry::ROTATION:
        transformDim = 3;
        rgbdEquationFuncPtr = calcRgbdEquationCoeffsRotation;
        icpEquationFuncPtr = calcICPEquationCoeffsRotation;
        // cout << "rotation" << rgbdEquationFuncPtr << " " << icpEquationFuncPtr << endl;
        break;
    case Odometry::TRANSLATION:
        transformDim = 3;
        rgbdEquationFuncPtr = calcRgbdEquationCoeffsTranslation;
        icpEquationFuncPtr = calcICPEquationCoeffsTranslation;
        // cout << "translation" << rgbdEquationFuncPtr << " " << icpEquationFuncPtr << endl;
        break;
    default:
        CV_Error(Error::StsBadArg, "Incorrect transformation type");
    }

    const int minOverdetermScale = 20;
    const int minCorrespsCount = minOverdetermScale * transformDim; //120
    const float icpWeight = 10.0;

    std::vector<Mat> pyramidCameraMatrix;
    
    buildPyramidCameraMatrix(cameraMatrix, (int)iterCounts.size(), pyramidCameraMatrix); // count = 4
    
    Mat resultRt = initRt.empty() ? Mat::eye(4,4,CV_64FC1) : initRt.clone();
    Mat currRt, ksi;
    // cout << "liyang test" << resultRt << endl;

    Mat resultRtPnP = initRt.empty() ? Mat::eye(4,4,CV_64FC1) : initRt.clone();
    Mat currRtPnP;
    for(int i=0; i<dstFrame->pyramidTexturedMask.size(); i++){
        // cout << dstFrame->pyramidTexturedMask[i].size() << " " << dstFrame->pyramidNormalsMask[i].size() << endl;
    }
    //cout << "liyang test" << srcFrame->pyramidDepth[1] << endl;
    //cout << "liyang test" << srcFrame->pyramidMask[1] << endl;
    //exit(1);
    bool isOk = false;
    double lastCost = 0;
    Sophus::SE3d pose;
    for(int level = (int)iterCounts.size() - 1; level >= 0; level--) // 3 2 1 0
    {
        cout << "level: " << level << " " << pyramidCameraMatrix[level] << endl;
        const Mat& levelCameraMatrix = pyramidCameraMatrix[level];
        const Mat& levelCameraMatrix_inv = levelCameraMatrix.inv(DECOMP_SVD);
        const Mat& srcLevelDepth = srcFrame->pyramidDepth[level];
        const Mat& dstLevelDepth = dstFrame->pyramidDepth[level];
        // cout << levelCameraMatrix << srcLevelDepth.rows << srcLevelDepth.cols; //60*80 120*160
        // cout << "+++++++++++++++++++++++++++++" << endl;
        const double fx = levelCameraMatrix.at<double>(0,0);
        const double fy = levelCameraMatrix.at<double>(1,1);
        const double determinantThreshold = 1e-6;

        Mat AtA_rgbd, AtB_rgbd, AtA_icp, AtB_icp;
        Mat corresps_rgbd, corresps_icp;

        vector<double> weight;
        Sophus::SE3d pose_gn;
        
        VecVector3d pts_3d_eigen;
        VecVector2d pts_2d_eigen;
        // Mat K = (Mat_<double>(3, 3) << 517.3f, 0, 318.6f, 0, 516.5f, 255.3f, 0, 0, 1);
        typedef Eigen::Matrix<double, 6, 1> Vector6d;
        Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
        Vector6d b = Vector6d::Zero();
        double cost = 0;
        // Run transformation search on current level iteratively.
        calculatePnPInliers(srcFrame->pyramidImage[level], dstFrame->pyramidImage[level], srcFrame->pyramidDepth[level], dstFrame->pyramidDepth[level],weight, pose_gn, pts_3d_eigen, pts_2d_eigen, resultRt, levelCameraMatrix, 0);
        for(int iter = 0; iter < iterCounts[level]; iter ++) // iter = 10 7 7 7
        {

            
            Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
            Vector6d b = Vector6d::Zero();
            cost = 0;
            Mat resultRt_inv = resultRt.inv(DECOMP_SVD);
            cout << resultRt_inv << endl;
            cout << "iter " << iter << endl;
            // cout << "iterRt " << resultRt << resultRtPnP << endl;
            // cout << "currRt " << currRt << currRtPnP;
            if(method & RGBD_ODOMETRY)

                computeCorresps(levelCameraMatrix, levelCameraMatrix_inv, resultRt_inv,
                                srcLevelDepth, srcFrame->pyramidMask[level], dstLevelDepth, dstFrame->pyramidTexturedMask[level],
                                maxDepthDiff, corresps_rgbd);

            if(method & ICP_ODOMETRY)
                computeCorresps(levelCameraMatrix, levelCameraMatrix_inv, resultRt_inv,
                                srcLevelDepth, srcFrame->pyramidMask[level], dstLevelDepth, dstFrame->pyramidNormalsMask[level],
                                maxDepthDiff, corresps_icp);
            // cout << corresps_rgbd << " " << corresps_icp << endl; // size: oo*4
            // if(corresps_rgbd.rows < minCorrespsCount && corresps_icp.rows < minCorrespsCount && iter > 0 && cost >= lastCost){
            //     cout << "too few" << endl;
            //     break;
            // }
            
            
            // check residual error convergence
            // cout << srcFrame->pyramidImage[level].size() << " " << srcFrame->pyramidDepth[level].size() << endl;
            cout << levelCameraMatrix << endl;
            

            Mat AtA(transformDim, transformDim, CV_64FC1, Scalar(0)), AtB(transformDim, 1, CV_64FC1, Scalar(0));
            if(corresps_rgbd.rows >= minCorrespsCount) // 120
            {
                calcRgbdLsmMatrices(srcFrame->pyramidImage[level], srcFrame->pyramidCloud[level], resultRt,
                                    dstFrame->pyramidImage[level], dstFrame->pyramid_dI_dx[level], dstFrame->pyramid_dI_dy[level],
                                    corresps_rgbd, fx, fy, sobelScale,
                                    AtA_rgbd, AtB_rgbd, rgbdEquationFuncPtr, transformDim);

                AtA += AtA_rgbd;
                AtB += AtB_rgbd;
            }
            if(corresps_icp.rows >= minCorrespsCount)
            {
                calcICPLsmMatrices(srcFrame->pyramidCloud[level], resultRt,
                                   dstFrame->pyramidCloud[level], dstFrame->pyramidNormals[level],
                                   corresps_icp, AtA_icp, AtB_icp, icpEquationFuncPtr, transformDim);
                //AtA += icpWeight * icpWeight * AtA_icp;
                //AtB += icpWeight * AtB_icp;
                AtA += AtA_icp;
                AtB += AtB_icp;
            }
            
            Mat K = levelCameraMatrix;
            double fx = K.at<double>(0, 0);
            double fy = K.at<double>(1, 1);
            double cx = K.at<double>(0, 2);
            double cy = K.at<double>(1, 2);
            Eigen::Matrix<double, 4, 4> Rt;
            // cv2eigen(resultRt, Rt);
            // cv2eigen(resultRtPnP, Rt);
            // cout << "Rt " << Rt << endl;
            for (int i = 0; i < pts_3d_eigen.size(); i++) {
                // cout << "pnp " << Rt << endl;
                // cout << (Rt * (pts_3d_eigen[i].homogeneous())).transpose().hnormalized().rows() << " " << (Rt * (pts_3d_eigen[i].homogeneous())).transpose().hnormalized().cols() << endl;
                // cout << (Rt * (pts_3d_eigen[i].homogeneous())).transpose().hnormalized() << endl;
                // for(int l=0; l<4; l++){
                //     cout << (pts_3d_eigen[i].homogeneous()).transpose()[l] << " ";
                // }
                // cout << endl;
                // Eigen::Vector3d pc = (Rt * (pts_3d_eigen[i].homogeneous())).transpose().hnormalized();
                Eigen::Vector3d pc = pose * pts_3d_eigen[i];
                double inv_z = 1.0 / pc[2];
                double inv_z2 = inv_z * inv_z;
                Eigen::Vector2d proj(fx * pc[0] / pc[2] + cx, fy * pc[1] / pc[2] + cy);
                Eigen::Vector2d e = pts_2d_eigen[i] - proj;
                // "[" << pts_3d_eigen[i][0] << ", " << pts_3d_eigen[i][1] << "]" <<"[" << pc[0] << ", " << pc[1] << "]" <<
                cout <<  "[" << pts_2d_eigen[i][0] << ", " << pts_2d_eigen[i][1] << "]" << " [" << proj[0] << ", " << proj[1] << "]" << endl;
                // cout << fx << " " << inv_z << " " << inv_z2 << " " << fx << " " << fy << endl;
                cost += e.squaredNorm();
                Eigen::Matrix<double, 2, 6> J;
                J << -fx * inv_z,
                    0,
                    fx * pc[0] * inv_z2,
                    fx * pc[0] * pc[1] * inv_z2,
                    -fx - fx * pc[0] * pc[0] * inv_z2,
                    fx * pc[1] * inv_z,
                    0,
                    -fy * inv_z,
                    fy * pc[1] * inv_z2,
                    fy + fy * pc[1] * pc[1] * inv_z2,
                    -fy * pc[0] * pc[1] * inv_z2,
                    -fy * pc[0] * inv_z;
                // cout << "J " << J << endl;
                // cout << endl;
                H += J.transpose() * (J * weight[i]);
                b += -J.transpose() * (e * weight[i]);
                // cout << J << endl;
                // cout << weight[i] << " " << e << endl;
            }
            cout << "cost " << cost << " " << lastCost << endl;
            cv::Mat AtA_pnp(cv::Size(6, 6), AtA_rgbd.type());
            cv::Mat AtB_pnp(cv::Size(1, 6), AtB_rgbd.type());
            // cout << AtA_pnp << " " << AtA_pnp.type() << endl;
            // cout << "ergeger    "  << H << endl;
            // cout << "ergeger    "  << b << endl;

            cv::eigen2cv(H, AtA_pnp);
            cv::eigen2cv(b, AtB_pnp);
            
            // cout << AtA_pnp << endl;
            // cout << AtB_pnp << endl;
            
            // // run only pnp method
            AtA += AtA_pnp;
            AtB += AtB_pnp;
            cout << "HHHHHHHHHH     " << AtA << endl;
            cout << "bbbbbbbbbb     " << AtB << endl;

            Eigen::Matrix<double, 6, 6> eigenH;
            Eigen::Matrix<double, 6, 1> eigenb;
            cv::cv2eigen(AtA, eigenH);
            cv::cv2eigen(AtB, eigenb);
            cout << "eigen " << eigenH << endl;
            cout << "eigen " << eigenb << endl;
            Vector6d dx;
            dx = eigenH.ldlt().solve(eigenb);
            if (dx.norm() < 1e-8) {
                // converge
                cout << "converge" << endl;
                break;
            }
            cout << "old pose " << pose.matrix() << endl;
            pose = Sophus::SE3d::exp(dx) * pose;
            cout << "new pose " << pose.matrix() << endl;

            bool solutionExist = solveSystem(AtA, AtB, determinantThreshold, ksi);
            if(!solutionExist){
                cout << "noSolution " << endl;
                break;
            }
            
            
            if(transfromType == Odometry::ROTATION)
            {
                Mat tmp(6, 1, CV_64FC1, Scalar(0));
                ksi.copyTo(tmp.rowRange(0,3));
                cout << "ksi1" << ksi << tmp << endl;
                ksi = tmp;
                cout << "ksi2" << ksi << endl;
            }
            else if(transfromType == Odometry::TRANSLATION)
            {
                Mat tmp(6, 1, CV_64FC1, Scalar(0));
                ksi.copyTo(tmp.rowRange(3,6));
                cout << "ksi3" << ksi << tmp << endl;
                ksi = tmp;
                cout << "ksi4" << ksi << endl;
            }
            cout << "ksi " << ksi << endl;
            if(iter > 0 && cost >= lastCost){
                cout << "cost: " << cost << ", last cost: " << lastCost << endl;
                break;
            }
            lastCost = cost;
            // computeProjectiveMatrixInv(ksi, currRtPnP);    
            // cout << "reverseRt " << currRtPnP << resultRtPnP << endl;
            // resultRtPnP = currRtPnP * resultRtPnP;

            computeProjectiveMatrix(ksi, currRt); // ksi size = 6*1
            // cout << "current" << currRt << resultRt << endl;
            resultRt = currRt * resultRt;
            isOk = true;
            
        }
        // cout << "result " << resultRt << endl;
    }

    Rt = resultRt;
    cout << Rt << Rt.size();
    cout << pose.matrix();
    Rt.at<double>(0,0) = pose.matrix()(0);
    Rt.at<double>(1,0) = pose.matrix()(1);
    Rt.at<double>(2,0) = pose.matrix()(2);
    Rt.at<double>(3,0) = pose.matrix()(3);
    Rt.at<double>(0,1) = pose.matrix()(4);
    Rt.at<double>(1,1) = pose.matrix()(5);
    Rt.at<double>(2,1) = pose.matrix()(6);
    Rt.at<double>(3,1) = pose.matrix()(7);
    Rt.at<double>(0,2) = pose.matrix()(8);
    Rt.at<double>(1,2) = pose.matrix()(9);
    Rt.at<double>(2,2) = pose.matrix()(10);
    Rt.at<double>(3,2) = pose.matrix()(11);
    Rt.at<double>(0,3) = pose.matrix()(12);
    Rt.at<double>(1,3) = pose.matrix()(13);
    Rt.at<double>(2,3) = pose.matrix()(14);
    Rt.at<double>(3,3) = pose.matrix()(15);
    cout << Rt << endl;
    if(isOk)
    {
        Mat deltaRt;

        if(initRt.empty())
            deltaRt = resultRt;
        else
            deltaRt = resultRt * initRt.inv(DECOMP_SVD);
        cout << "initRt " << initRt << " " << deltaRt << endl;
        isOk = testDeltaTransformation(deltaRt, maxTranslation, maxRotation);
    }
    cout << "isOk" << isOk << endl;
    return isOk;
}

template<class ImageElemType>
static void
warpFrameImpl(const Mat& image, const Mat& depth, const Mat& mask,
              const Mat& Rt, const Mat& cameraMatrix, const Mat& distCoeff,
              Mat& warpedImage, Mat* warpedDepth, Mat* warpedMask)
{
    CV_Assert(image.size() == depth.size());

    Mat cloud;
    depthTo3d(depth, cameraMatrix, cloud);

    std::vector<Point2f> points2d;
    Mat transformedCloud;
    perspectiveTransform(cloud, transformedCloud, Rt);
    projectPoints(transformedCloud.reshape(3, 1), Mat::eye(3, 3, CV_64FC1), Mat::zeros(3, 1, CV_64FC1), cameraMatrix,
                distCoeff, points2d);

    warpedImage = Mat(image.size(), image.type(), Scalar::all(0));

    Mat zBuffer(image.size(), CV_32FC1, std::numeric_limits<float>::max());
    const Rect rect = Rect(0, 0, image.cols, image.rows);

    for (int y = 0; y < image.rows; y++)
    {
        //const Point3f* cloud_row = cloud.ptr<Point3f>(y);
        const Point3f* transformedCloud_row = transformedCloud.ptr<Point3f>(y);
        const Point2f* points2d_row = &points2d[y*image.cols];
        const ImageElemType* image_row = image.ptr<ImageElemType>(y);
        const uchar* mask_row = mask.empty() ? 0 : mask.ptr<uchar>(y);
        for (int x = 0; x < image.cols; x++)
        {
            const float transformed_z = transformedCloud_row[x].z;
            const Point2i p2d = points2d_row[x];
            if((!mask_row || mask_row[x]) && transformed_z > 0 && rect.contains(p2d) && /*!cvIsNaN(cloud_row[x].z) && */zBuffer.at<float>(p2d) > transformed_z)
            {
                warpedImage.at<ImageElemType>(p2d) = image_row[x];
                zBuffer.at<float>(p2d) = transformed_z;
            }
        }
    }

    if(warpedMask)
        *warpedMask = zBuffer != std::numeric_limits<float>::max();

    if(warpedDepth)
    {
        zBuffer.setTo(std::numeric_limits<float>::quiet_NaN(), zBuffer == std::numeric_limits<float>::max());
        *warpedDepth = zBuffer;
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////

RgbdFrame::RgbdFrame() : ID(-1)
{}

RgbdFrame::RgbdFrame(const Mat& image_in, const Mat& depth_in, const Mat& mask_in, const Mat& normals_in, int ID_in)
    : ID(ID_in), image(image_in), depth(depth_in), mask(mask_in), normals(normals_in)
{}

RgbdFrame::~RgbdFrame()
{}

void RgbdFrame::release()
{
    ID = -1;
    image.release();
    depth.release();
    mask.release();
    normals.release();
}

OdometryFrame::OdometryFrame() : RgbdFrame()
{}

OdometryFrame::OdometryFrame(const Mat& image_in, const Mat& depth_in, const Mat& mask_in, const Mat& normals_in, int ID_in)
    : RgbdFrame(image_in, depth_in, mask_in, normals_in, ID_in)
{}

void OdometryFrame::release()
{
    RgbdFrame::release();
    releasePyramids();
}

void OdometryFrame::releasePyramids()
{
    pyramidImage.clear();
    pyramidDepth.clear();
    pyramidMask.clear();

    pyramidCloud.clear();

    pyramid_dI_dx.clear();
    pyramid_dI_dy.clear();
    pyramidTexturedMask.clear();

    pyramidNormals.clear();
    pyramidNormalsMask.clear();
}

bool Odometry::compute(const Mat& srcImage, const Mat& srcDepth, const Mat& srcMask,
                       const Mat& dstImage, const Mat& dstDepth, const Mat& dstMask,
                       Mat& Rt, const Mat& initRt) const
{
    Ptr<OdometryFrame> srcFrame(new OdometryFrame(srcImage, srcDepth, srcMask));
    Ptr<OdometryFrame> dstFrame(new OdometryFrame(dstImage, dstDepth, dstMask));

    return compute(srcFrame, dstFrame, Rt, initRt);
}

bool Odometry::compute(Ptr<OdometryFrame>& srcFrame, Ptr<OdometryFrame>& dstFrame, Mat& Rt, const Mat& initRt) const
{
    checkParams();

    Size srcSize = prepareFrameCache(srcFrame, OdometryFrame::CACHE_SRC);
    Size dstSize = prepareFrameCache(dstFrame, OdometryFrame::CACHE_DST);
    cout << srcSize << dstSize << endl;
    if(srcSize != dstSize)
        CV_Error(Error::StsBadSize, "srcFrame and dstFrame have to have the same size (resolution).");
    // cout << "compute here" << endl;
    // cout << "compute " << srcFrame->depth.type() << " " << dstFrame->depth.type() << endl;
    return computeImpl(srcFrame, dstFrame, Rt, initRt);
}

Size Odometry::prepareFrameCache(Ptr<OdometryFrame> &frame, int /*cacheType*/) const
{
    cout << "1" << endl;
    if(frame == 0)
        CV_Error(Error::StsBadArg, "Null frame pointer.\n");

    return Size();
}

Ptr<Odometry> Odometry::create(const String & odometryType)
{
    if (odometryType == "RgbdOdometry")
        return makePtr<RgbdOdometry>();
    else if (odometryType == "ICPOdometry")
        return makePtr<ICPOdometry>();
    else if (odometryType == "RgbdICPOdometry")
        return makePtr<RgbdICPOdometry>();
    return Ptr<Odometry>();
}

//
RgbdOdometry::RgbdOdometry() :
    minDepth(DEFAULT_MIN_DEPTH()),
    maxDepth(DEFAULT_MAX_DEPTH()),
    maxDepthDiff(DEFAULT_MAX_DEPTH_DIFF()),
    maxPointsPart(DEFAULT_MAX_POINTS_PART()),
    transformType(Odometry::RIGID_BODY_MOTION),
    maxTranslation(DEFAULT_MAX_TRANSLATION()),
    maxRotation(DEFAULT_MAX_ROTATION())

{
    setDefaultIterCounts(iterCounts);
    setDefaultMinGradientMagnitudes(minGradientMagnitudes);
}

RgbdOdometry::RgbdOdometry(const Mat& _cameraMatrix,
                           float _minDepth, float _maxDepth, float _maxDepthDiff,
                           const std::vector<int>& _iterCounts,
                           const std::vector<float>& _minGradientMagnitudes,
                           float _maxPointsPart,
                           int _transformType) :
                           minDepth(_minDepth), maxDepth(_maxDepth), maxDepthDiff(_maxDepthDiff),
                           iterCounts(Mat(_iterCounts).clone()),
                           minGradientMagnitudes(Mat(_minGradientMagnitudes).clone()),
                           maxPointsPart(_maxPointsPart),
                           cameraMatrix(_cameraMatrix), transformType(_transformType),
                           maxTranslation(DEFAULT_MAX_TRANSLATION()), maxRotation(DEFAULT_MAX_ROTATION())
{
    if(iterCounts.empty() || minGradientMagnitudes.empty())
    {
        setDefaultIterCounts(iterCounts);
        setDefaultMinGradientMagnitudes(minGradientMagnitudes);
    }
}

Size RgbdOdometry::prepareFrameCache(Ptr<OdometryFrame>& frame, int cacheType) const
{
    Odometry::prepareFrameCache(frame, cacheType);
    cout << "2" << endl;
    if(frame->image.empty())
    {
        if(!frame->pyramidImage.empty())
            frame->image = frame->pyramidImage[0];
        else
            CV_Error(Error::StsBadSize, "Image or pyramidImage have to be set.");
    }
    checkImage(frame->image);

    if(frame->depth.empty())
    {
        if(!frame->pyramidDepth.empty()){
            frame->depth = frame->pyramidDepth[0];
            // cout << "          if " << frame->image.channels() << endl;
            // cout << "          if " << frame->depth.channels() << endl;
        }
        else if(!frame->pyramidCloud.empty())
        {
            Mat cloud = frame->pyramidCloud[0];
            std::vector<Mat> xyz;
            split(cloud, xyz);
            frame->depth = xyz[2];
            // cout << "else          if " << frame->image.channels() << endl;
            // cout << "else          if " << frame->depth.channels() << endl;
        }
        else
            CV_Error(Error::StsBadSize, "Depth or pyramidDepth or pyramidCloud have to be set.");
    }
    checkDepth(frame->depth, frame->image.size());
    // cout << "else          if " << frame->image.channels() << endl;
    // cout << "else          if " << frame->depth.channels() << endl;
    if(frame->mask.empty() && !frame->pyramidMask.empty())
        frame->mask = frame->pyramidMask[0];
    checkMask(frame->mask, frame->image.size());

    preparePyramidImage(frame->image, frame->pyramidImage, iterCounts.total());
    // cout << frame->pyramidImage[3].channels() << frame->pyramidImage[3].type() << endl;
    preparePyramidDepth(frame->depth, frame->pyramidDepth, iterCounts.total());
    // cout << frame->pyramidDepth[3].channels() << frame->pyramidDepth[3].type() << endl;
    preparePyramidMask(frame->mask, frame->pyramidDepth, (float)minDepth, (float)maxDepth,
                       frame->pyramidNormals, frame->pyramidMask);

    if(cacheType & OdometryFrame::CACHE_SRC)
        preparePyramidCloud(frame->pyramidDepth, cameraMatrix, frame->pyramidCloud);

    if(cacheType & OdometryFrame::CACHE_DST)
    {
        preparePyramidSobel(frame->pyramidImage, 1, 0, frame->pyramid_dI_dx);
        preparePyramidSobel(frame->pyramidImage, 0, 1, frame->pyramid_dI_dy);
        preparePyramidTexturedMask(frame->pyramid_dI_dx, frame->pyramid_dI_dy, minGradientMagnitudes,
                                   frame->pyramidMask, maxPointsPart, frame->pyramidTexturedMask);
    }

    return frame->image.size();
}

void RgbdOdometry::checkParams() const
{
    CV_Assert(maxPointsPart > 0. && maxPointsPart <= 1.);
    CV_Assert(cameraMatrix.size() == Size(3,3) && (cameraMatrix.type() == CV_32FC1 || cameraMatrix.type() == CV_64FC1));
    CV_Assert(minGradientMagnitudes.size() == iterCounts.size() || minGradientMagnitudes.size() == iterCounts.t().size());
}

bool RgbdOdometry::computeImpl(const Ptr<OdometryFrame>& srcFrame, const Ptr<OdometryFrame>& dstFrame, Mat& Rt, const Mat& initRt) const
{
    return RGBDICPOdometryImpl(Rt, initRt, srcFrame, dstFrame, cameraMatrix, (float)maxDepthDiff, iterCounts, maxTranslation, maxRotation, RGBD_ODOMETRY, transformType);
}

//
ICPOdometry::ICPOdometry() :
    minDepth(DEFAULT_MIN_DEPTH()), maxDepth(DEFAULT_MAX_DEPTH()),
    maxDepthDiff(DEFAULT_MAX_DEPTH_DIFF()), maxPointsPart(DEFAULT_MAX_POINTS_PART()), transformType(Odometry::RIGID_BODY_MOTION),
    maxTranslation(DEFAULT_MAX_TRANSLATION()), maxRotation(DEFAULT_MAX_ROTATION())
{
    setDefaultIterCounts(iterCounts);
}

ICPOdometry::ICPOdometry(const Mat& _cameraMatrix,
                         float _minDepth, float _maxDepth, float _maxDepthDiff,
                         float _maxPointsPart, const std::vector<int>& _iterCounts,
                         int _transformType) :
                         minDepth(_minDepth), maxDepth(_maxDepth), maxDepthDiff(_maxDepthDiff),
                         maxPointsPart(_maxPointsPart), iterCounts(Mat(_iterCounts).clone()),
                         cameraMatrix(_cameraMatrix), transformType(_transformType),
                         maxTranslation(DEFAULT_MAX_TRANSLATION()), maxRotation(DEFAULT_MAX_ROTATION())
{
    if(iterCounts.empty())
        setDefaultIterCounts(iterCounts);
}

Size ICPOdometry::prepareFrameCache(Ptr<OdometryFrame>& frame, int cacheType) const
{
    Odometry::prepareFrameCache(frame, cacheType);
    cout << "3" << endl;
    if(frame->depth.empty())
    {
        if(!frame->pyramidDepth.empty())
            frame->depth = frame->pyramidDepth[0];
        else if(!frame->pyramidCloud.empty())
        {
            Mat cloud = frame->pyramidCloud[0];
            std::vector<Mat> xyz;
            split(cloud, xyz);
            frame->depth = xyz[2];
        }
        else
            CV_Error(Error::StsBadSize, "Depth or pyramidDepth or pyramidCloud have to be set.");
    }
    checkDepth(frame->depth, frame->depth.size());

    if(frame->mask.empty() && !frame->pyramidMask.empty())
        frame->mask = frame->pyramidMask[0];
    checkMask(frame->mask, frame->depth.size());

    preparePyramidDepth(frame->depth, frame->pyramidDepth, iterCounts.total());

    preparePyramidCloud(frame->pyramidDepth, cameraMatrix, frame->pyramidCloud);

    if(cacheType & OdometryFrame::CACHE_DST)
    {
        if(frame->normals.empty())
        {
            if(!frame->pyramidNormals.empty())
                frame->normals = frame->pyramidNormals[0];
            else
            {
                if(normalsComputer.empty() ||
                   normalsComputer->getRows() != frame->depth.rows ||
                   normalsComputer->getCols() != frame->depth.cols ||
                   norm(normalsComputer->getK(), cameraMatrix) > FLT_EPSILON)
                   normalsComputer = makePtr<RgbdNormals>(frame->depth.rows,
                                                          frame->depth.cols,
                                                          frame->depth.depth(),
                                                          cameraMatrix,
                                                          normalWinSize,
                                                          normalMethod);

                (*normalsComputer)(frame->pyramidCloud[0], frame->normals);
            }
        }
        checkNormals(frame->normals, frame->depth.size());

        preparePyramidNormals(frame->normals, frame->pyramidDepth, frame->pyramidNormals);

        preparePyramidMask(frame->mask, frame->pyramidDepth, (float)minDepth, (float)maxDepth,
                           frame->pyramidNormals, frame->pyramidMask);

        preparePyramidNormalsMask(frame->pyramidNormals, frame->pyramidMask, maxPointsPart, frame->pyramidNormalsMask);
    }
    else
        preparePyramidMask(frame->mask, frame->pyramidDepth, (float)minDepth, (float)maxDepth,
                           frame->pyramidNormals, frame->pyramidMask);

    return frame->depth.size();
}

void ICPOdometry::checkParams() const
{
    CV_Assert(maxPointsPart > 0. && maxPointsPart <= 1.);
    CV_Assert(cameraMatrix.size() == Size(3,3) && (cameraMatrix.type() == CV_32FC1 || cameraMatrix.type() == CV_64FC1));
}

bool ICPOdometry::computeImpl(const Ptr<OdometryFrame>& srcFrame, const Ptr<OdometryFrame>& dstFrame, Mat& Rt, const Mat& initRt) const
{
    return RGBDICPOdometryImpl(Rt, initRt, srcFrame, dstFrame, cameraMatrix, (float)maxDepthDiff, iterCounts, maxTranslation, maxRotation, ICP_ODOMETRY, transformType);
}

//
RgbdICPOdometry::RgbdICPOdometry() :
    minDepth(DEFAULT_MIN_DEPTH()), maxDepth(DEFAULT_MAX_DEPTH()),
    maxDepthDiff(DEFAULT_MAX_DEPTH_DIFF()), maxPointsPart(DEFAULT_MAX_POINTS_PART()), transformType(Odometry::RIGID_BODY_MOTION),
    maxTranslation(DEFAULT_MAX_TRANSLATION()), maxRotation(DEFAULT_MAX_ROTATION())
{
    setDefaultIterCounts(iterCounts);
    setDefaultMinGradientMagnitudes(minGradientMagnitudes);
}

RgbdICPOdometry::RgbdICPOdometry(const Mat& _cameraMatrix,
                                 float _minDepth, float _maxDepth, float _maxDepthDiff,
                                 float _maxPointsPart, const std::vector<int>& _iterCounts,
                                 const std::vector<float>& _minGradientMagnitudes,
                                 int _transformType) :
                                 minDepth(_minDepth), maxDepth(_maxDepth), maxDepthDiff(_maxDepthDiff),
                                 maxPointsPart(_maxPointsPart), iterCounts(Mat(_iterCounts).clone()),
                                 minGradientMagnitudes(Mat(_minGradientMagnitudes).clone()),
                                 cameraMatrix(_cameraMatrix), transformType(_transformType),
                                 maxTranslation(DEFAULT_MAX_TRANSLATION()), maxRotation(DEFAULT_MAX_ROTATION())
{
    if(iterCounts.empty() || minGradientMagnitudes.empty())
    {
        setDefaultIterCounts(iterCounts);
        setDefaultMinGradientMagnitudes(minGradientMagnitudes);
    }
}

Size RgbdICPOdometry::prepareFrameCache(Ptr<OdometryFrame>& frame, int cacheType) const
{
    cout << "4 " << frame->image.empty() << " " << frame->depth.empty() << " " << frame->mask.empty() << endl;
    if(frame->image.empty())
    {
        if(!frame->pyramidImage.empty()){
            frame->image = frame->pyramidImage[0];
            // for(int i=0; i<4; i++){
            //     cout << "index1 " << i << " " << frame->pyramidImage[i] << endl;
            // }
        }
        else
            CV_Error(Error::StsBadSize, "Image or pyramidImage have to be set.");
    }
    checkImage(frame->image);

    if(frame->depth.empty())
    {
        if(!frame->pyramidDepth.empty()){
            frame->depth = frame->pyramidDepth[0];
            for(int i=0; i<4; i++){
                // cout << "index2 " << i << " " << frame->pyramidDepth[i].size() << endl;
            }
        }
        else if(!frame->pyramidCloud.empty())
        {
            Mat cloud = frame->pyramidCloud[0];
            for(int i=0; i<4; i++){
                // cout << "index3 " << i << " " << frame->pyramidCloud[i].size() << endl;
            }
            std::vector<Mat> xyz;
            split(cloud, xyz);
            frame->depth = xyz[2];
            // cout << "xyz size" << xyz.size() << endl;
        }
        else
            CV_Error(Error::StsBadSize, "Depth or pyramidDepth or pyramidCloud have to be set.");
    }
    checkDepth(frame->depth, frame->image.size());

    if(frame->mask.empty() && !frame->pyramidMask.empty()){
        frame->mask = frame->pyramidMask[0];
        for(int i=0; i<4; i++){
            // cout << "index4 " << i << " " << frame->pyramidMask[i].size() << endl;
        }
    }
    checkMask(frame->mask, frame->image.size());
    // cout << frame->image.size() << " " << frame->depth.size() << " " << frame->pyramidImage.size() << " " << frame->pyramidDepth.size() << " " << endl;
    // cout << frame->pyramidCloud.size() << " " << frame->pyramidNormals.size() << " " << frame->normals.size() << " " << frame->pyramidMask.size() << " " << endl;
    // cout << frame->mask.size() << " " << iterCounts.total() << endl;
    // cout << frame->pyramid_dI_dx.size() << " " << frame->pyramidTexturedMask.size() << endl;
    preparePyramidImage(frame->image, frame->pyramidImage, iterCounts.total());

    preparePyramidDepth(frame->depth, frame->pyramidDepth, iterCounts.total());

    preparePyramidCloud(frame->pyramidDepth, cameraMatrix, frame->pyramidCloud);

    //cout << "liyang test" << frame->mask << endl;
    //exit(1);
    cout << frame->pyramidMask.size() << " " << frame->mask.size() << " " << frame->mask.empty() << " " << frame->pyramidMask.empty() << endl;
    for(int i=0; i<frame->pyramidImage.size(); i++){
        // cout << "index5 " << i << " " << frame->pyramidImage[i].size() << " " << frame->pyramidDepth[i].size() << " " << frame->pyramidCloud[i].size() << " "  << endl;
    }
    if(cacheType & OdometryFrame::CACHE_DST)
    {
        if(frame->normals.empty())
        {
            // cout << "hi " << frame->normals.size() << " " << frame->pyramidNormals.size() << endl;

            if(!frame->pyramidNormals.empty()){
                frame->normals = frame->pyramidNormals[0];
                // cout << "a" << endl;
            }
            else
            {
                // cout << "b" << endl;
                if(normalsComputer.empty() ||
                   normalsComputer->getRows() != frame->depth.rows ||
                   normalsComputer->getCols() != frame->depth.cols ||
                   norm(normalsComputer->getK(), cameraMatrix) > FLT_EPSILON)
                   normalsComputer = makePtr<RgbdNormals>(frame->depth.rows,
                                                          frame->depth.cols,
                                                          frame->depth.depth(),
                                                          cameraMatrix,
                                                          normalWinSize,
                                                          normalMethod);

                (*normalsComputer)(frame->pyramidCloud[0], frame->normals);
            }
        }
        // cout << "hi " << frame->normals.size() << " " << frame->pyramidNormals.size() << endl;
        // cout << "hey " << normalWinSize << " " << normalMethod << endl;
        checkNormals(frame->normals, frame->depth.size());

        preparePyramidNormals(frame->normals, frame->pyramidDepth, frame->pyramidNormals);

        preparePyramidMask(frame->mask, frame->pyramidDepth, (float)minDepth, (float)maxDepth,
                           frame->pyramidNormals, frame->pyramidMask);

        //cout << "liyang test" << frame->mask << endl;
        //exit(1);
        preparePyramidSobel(frame->pyramidImage, 1, 0, frame->pyramid_dI_dx);
        preparePyramidSobel(frame->pyramidImage, 0, 1, frame->pyramid_dI_dy);
        preparePyramidTexturedMask(frame->pyramid_dI_dx, frame->pyramid_dI_dy,
                                   minGradientMagnitudes, frame->pyramidMask,
                                   maxPointsPart, frame->pyramidTexturedMask);

        preparePyramidNormalsMask(frame->pyramidNormals, frame->pyramidMask, maxPointsPart, frame->pyramidNormalsMask);
    }
    else{
        // cout << frame->pyramidMask.size() << " " << frame->mask.size() << " " << frame->mask.empty() << " " << frame->pyramidMask.empty() << endl;
        preparePyramidMask(frame->mask, frame->pyramidDepth, (float)minDepth, (float)maxDepth,
                           frame->pyramidNormals, frame->pyramidMask);
    }
    // cout << frame->image.size() << " " << frame->depth.size() << " " << frame->pyramidImage.size() << " " << frame->pyramidDepth.size() << " " << endl;
    // cout << frame->pyramidCloud.size() << " " << frame->pyramidNormals.size() << " " << frame->normals.size() << " " << frame->pyramidMask.size() << " " << endl;
    // cout << frame->mask.size() << " " << iterCounts.total() << endl;
    // cout << frame->pyramid_dI_dx.size() << " " << frame->pyramidTexturedMask.size() << endl;
    return frame->image.size();
}

void RgbdICPOdometry::checkParams() const
{
    CV_Assert(maxPointsPart > 0. && maxPointsPart <= 1.);
    CV_Assert(cameraMatrix.size() == Size(3,3) && (cameraMatrix.type() == CV_32FC1 || cameraMatrix.type() == CV_64FC1));
    CV_Assert(minGradientMagnitudes.size() == iterCounts.size() || minGradientMagnitudes.size() == iterCounts.t().size());
}

bool RgbdICPOdometry::computeImpl(const Ptr<OdometryFrame>& srcFrame, const Ptr<OdometryFrame>& dstFrame, Mat& Rt, const Mat& initRt) const
{
    return RGBDICPOdometryImpl(Rt, initRt, srcFrame, dstFrame, cameraMatrix, (float)maxDepthDiff, iterCounts,  maxTranslation, maxRotation, MERGED_ODOMETRY, transformType);
}

//

void
warpFrame(const Mat& image, const Mat& depth, const Mat& mask,
          const Mat& Rt, const Mat& cameraMatrix, const Mat& distCoeff,
          Mat& warpedImage, Mat* warpedDepth, Mat* warpedMask)
{
    if(image.type() == CV_8UC1)
        warpFrameImpl<uchar>(image, depth, mask, Rt, cameraMatrix, distCoeff, warpedImage, warpedDepth, warpedMask);
    else if(image.type() == CV_8UC3)
        warpFrameImpl<Point3_<uchar> >(image, depth, mask, Rt, cameraMatrix, distCoeff, warpedImage, warpedDepth, warpedMask);
    else
        CV_Error(Error::StsBadArg, "Image has to be type of CV_8UC1 or CV_8UC3");
}
}
} // namespace cv
