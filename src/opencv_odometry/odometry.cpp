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
typedef Eigen::Matrix<double, 6, 1> Vector6d;
vector<int> nlevels{8};
vector<int> patchSizes{31};
vector<int> edgeThresholds = patchSizes;

void ransac_matching(
  vector<DMatch> &match_inliers, 
  vector<DMatch> &matches, 
  const int minNumberMatchesAllowed,
  vector<KeyPoint> &keypoints_prev, 
  vector<KeyPoint> &keypoints_curr)
  {

    if (matches.size() > minNumberMatchesAllowed){
        // Prepare data for cv::findHomography
      std::vector<cv::Point2f> srcPoints(matches.size());
      std::vector<cv::Point2f> dstPoints(matches.size());

      // for (DMatch m:matches) {
      for (size_t i = 0; i < matches.size(); i++){
          srcPoints[i] = keypoints_prev[matches[i].queryIdx].pt;
          dstPoints[i] = keypoints_curr[matches[i].trainIdx].pt;
      }
      std::vector<unsigned char> inliersMask(srcPoints.size());
      Mat homography = cv::findHomography(srcPoints, dstPoints, RANSAC, 2,inliersMask, 2000, 0.995);
      
      for (size_t l=0; l<inliersMask.size(); l++){
          if (inliersMask[l])
              match_inliers.push_back(matches[l]);
      }
      cout << "inlier size " << match_inliers.size() << endl;
    } else {
      match_inliers = matches;
    }
}
void find_feature_matches_orb(const Mat &img_1, const Mat &img_2,
                              vector<KeyPoint> &keypoints_1,
			                  vector<KeyPoint> &keypoints_2,
                              vector<DMatch> &matches,
                              int level) {

    Mat descriptors_1, descriptors_2;
    int nfeatures = 1000;
    Ptr<ORB> detector = ORB::create(nfeatures);
    detector->detectAndCompute(img_1, noArray(), keypoints_1, descriptors_1);
    detector->detectAndCompute(img_2, noArray(), keypoints_2, descriptors_2);
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("FlannBased");
    matcher->clear();
    vector<DMatch> match;
    vector<vector<DMatch>> knn_matches; 
    descriptors_1.convertTo(descriptors_1, CV_32F);
    descriptors_2.convertTo(descriptors_2, CV_32F);
    matcher->knnMatch( descriptors_1, descriptors_2, knn_matches, 2);
    cout << "knn " << knn_matches.size() << " " << matches.size() << " " << descriptors_1.type() << " " << descriptors_2.type() << " " << img_1.type() << " " << img_2.type() << endl;
    cout << "key " << matches.size() << " " << keypoints_1.size() << " " << keypoints_2.size() << endl;
    const float ratio_thresh = 0.7f;
    for (size_t i = 0; i < knn_matches.size(); i++){
        if(knn_matches[i].size() >= 2){
            if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance){
                matches.push_back(knn_matches[i][0]);
            }
        }
    }
}

double Average(vector<double> v)
{      double sum=0;
       for(int i=0;i<v.size();i++)
               sum+=v[i];
       return sum/v.size();
       // check if average converges
}
double Deviation(vector<double> v, double ave)
{
       double E=0;
       for(int i=0;i<v.size();i++){
               E+=(v[i] - ave)*(v[i] - ave);
       }
       return sqrt(E/v.size());
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
    const Mat& resultRt
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
void match_points(
  vector<Point3f> &pts_3d, 
  vector<Point2f> &pts_2d, 
  vector<Point3f> &pts_3d_nxt, 
  vector<KeyPoint> &keys1, 
  vector<KeyPoint> &keys2,
  vector<KeyPoint> &keypoints_prev, 
  vector<KeyPoint> &keypoints_curr,
  vector<Point2f> &points1, 
  vector<Point2f> &points2, 
  vector<DMatch> &matches,
  const Mat &prev_depth, 
  const Mat &curr_depth,
  const Mat &K){

  int index = 1;
  cout << "match size " << matches.size() << endl;
  for (DMatch m:matches) {
    float d = prev_depth.ptr<float>(int(keypoints_prev[m.queryIdx].pt.y))[int(keypoints_prev[m.queryIdx].pt.x)];
    float d_nxt = curr_depth.ptr<float>(int(keypoints_curr[m.trainIdx].pt.y))[int(keypoints_curr[m.trainIdx].pt.x)];
    index += 1;
    if (d == 0 || d_nxt == 0 || isnan(d) || isnan(d_nxt)){
        continue;
    }
    float dd = d;
    float dd_nxt = d_nxt;
    Point2d p1 = pixel2cam(keypoints_prev[m.queryIdx].pt, K);
    Point2d p2 = pixel2cam(keypoints_curr[m.trainIdx].pt, K);
    // cout << "(" << keypoints_prev[m.queryIdx].pt.x << "," << keypoints_prev[m.queryIdx].pt.y << ") (" << keypoints_curr[m.trainIdx].pt.x << "," << keypoints_curr[m.trainIdx].pt.y << ") " << dd << " " << dd_nxt << "(" << p1.x << "," << p1.y << ") (" << p2.x << "," << p2.y << ")" << endl;
    keys1.push_back(keypoints_prev[m.queryIdx]);
    keys2.push_back(keypoints_curr[m.trainIdx]);
    points1.push_back(keypoints_prev[m.queryIdx].pt);
    points2.push_back(keypoints_curr[m.trainIdx].pt);
    pts_3d.push_back(Point3f(p1.x * dd, p1.y * dd, dd));
    pts_3d_nxt.push_back(Point3f(p2.x * dd_nxt, p2.y * dd_nxt, dd_nxt));
    pts_2d.push_back(keypoints_curr[m.trainIdx].pt);
  }
}
void calculatePnPInliers(const Mat& frame0,
                         const Mat& frame1,
                         const Mat& depth0,
                         const Mat& depth1,
                         vector<double>& weight,
                         Sophus::SE3d& pose_gn,
                         VecVector3d& pts_3d_eigen,
                         VecVector2d& pts_2d_eigen,
                         const Mat& resultRt,
                         const Mat& K,
                         int level
                         ){
    
    CV_Assert(!depth0.empty());
    CV_Assert(!depth1.empty());
    
    std::vector<KeyPoint> keypoints_prev, keypoints_curr, key_temp1, key_temp2;
    Ptr<ORB> detector = ORB::create(1000);
    detector->setNLevels(nlevels[level]);
    detector->setEdgeThreshold(edgeThresholds[level]);
    detector->setPatchSize(patchSizes[level]);
    detector->detect(frame0, key_temp1);
    detector->detect(frame1, key_temp2);
    cout << nlevels[level] <<  " " <<  edgeThresholds[level] << " " << patchSizes[level] << " " << key_temp1.size() << " " << key_temp2.size() << " " << frame0.type() << " " << frame1.type() << endl;
    vector<DMatch> matches;
    cout << matches.size() << endl;
    keypoints_prev.clear();
    keypoints_curr.clear();
    find_feature_matches_orb(frame0, frame1, keypoints_prev, keypoints_curr, matches, level);
    cout << "第一個: " <<  "一共找到了" << keypoints_prev.size() << " " << keypoints_curr.size() << "個關鍵點" << endl;
    cout << "第一個: " << "一共找到了" << matches.size() << "组匹配点" << endl;
    const int minNumberMatchesAllowed = 8;
    std::vector<cv::DMatch> match_inliers;
    ransac_matching(match_inliers, matches, minNumberMatchesAllowed, keypoints_prev, keypoints_curr);
    
    vector<KeyPoint> keys1, keys2;
    vector<Point3f> pts_3d;
    vector<Point2f> pts_2d;
    vector<Point3f> pts_3d_nxt;
    vector<Point2f> points1, points2;

    match_points(pts_3d, pts_2d, pts_3d_nxt, keys1, keys2, keypoints_prev, keypoints_curr, points1, points2, match_inliers, depth0, depth1, K);
    cout << "3d-2d pairs: " << pts_3d.size() << " " << pts_2d.size() << endl;
    
    VecVector3d pts_3d_eigen_nxt;
    for (size_t i = 0; i < pts_3d.size(); ++i) {
        // cout << "vector3d " << Eigen::Vector3d(pts_3d[i].x, pts_3d[i].y, pts_3d[i].z) << "vector2d " << Eigen::Vector2d(pts_2d[i].x, pts_2d[i].y) << endl;
        pts_3d_eigen.push_back(Eigen::Vector3d(pts_3d[i].x, pts_3d[i].y, pts_3d[i].z));
        pts_3d_eigen_nxt.push_back(Eigen::Vector3d(pts_3d_nxt[i].x, pts_3d_nxt[i].y, pts_3d_nxt[i].z));
        pts_2d_eigen.push_back(Eigen::Vector2d(pts_2d[i].x, pts_2d[i].y));
    }
    keypoints_prev.clear();
    keypoints_prev = keys1;
    keypoints_curr.clear();
    keypoints_curr = keys2;
    Mat r, t, inliers, Rt_ba;;
    cout << "3d-2d pairs: " << pts_3d.size() << " " << pts_2d.size() << endl;
    int mode = 0; // 0=huber
    vector<double> residuals;
    double res_std = calc_residual(pts_3d_eigen, pts_2d_eigen, pose_gn, K, residuals, resultRt);
    cout << "deviation: " << res_std << endl;
    double huber_k = 1.345 * res_std;
    if(mode == 0){
        for (int j=0; j<residuals.size(); j++){
            if(residuals[j] <= huber_k){
                weight.push_back(1.0);
            }else {
                weight.push_back(huber_k/residuals[j]);
            }

        }
    }else {
        for (int j=0; j<residuals.size(); j++){
            weight.push_back(0.0);
        }
    }
}
    

static inline
void setDefaultIterCounts(Mat& iterCounts)
{
    iterCounts = Mat(Vec4i(7,7,7,10));
    // iterCounts = Mat(Vec<int,1>(10));
}

static inline
void setDefaultMinGradientMagnitudes(Mat& minGradientMagnitudes)
{
    minGradientMagnitudes = Mat(Vec4f(10,10,10,10));
    // minGradientMagnitudes = Mat(Vec<float,1>(10));
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
            CV_Error(Error::StsBadSize, "Mask type has to be CV_8UC1."); // mask type 8UC1
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
void preparePyramidImage(const Mat& image, std::vector<Mat>& pyramidImage, size_t levelCount) // levelCount=4
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
                        std::vector<Mat>& pyramidMask) // minDepth=0, maxDepth=4
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

        buildPyramid(validMask, pyramidMask, (int)pyramidDepth.size() - 1);
        for(size_t i = 0; i < pyramidMask.size(); i++) // pyramidMask size=4
        {
            Mat levelDepth = pyramidDepth[i].clone();
            patchNaNs(levelDepth, 0);
            Mat& levelMask = pyramidMask[i];
            levelMask &= (levelDepth > minDepth) & (levelDepth < maxDepth); // correct
            // levelMask &= (levelDepth > minDepth); // fix
            if(!pyramidNormal.empty()) // pyranmidNormal type 32FC3
            {
                CV_Assert(pyramidNormal[i].type() == CV_32FC3);
                CV_Assert(pyramidNormal[i].size() == pyramidDepth[i].size());
                Mat levelNormal = pyramidNormal[i].clone();
                Mat validNormalMask = levelNormal == levelNormal; // otherwise it's Nan
                CV_Assert(validNormalMask.type() == CV_8UC3);
                // cout << pyramidNormal[i].type() << " " << levelNormal.type() << " " << validNormalMask.type() << endl;

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
            CV_Assert(pyramidCloud[i].type() == CV_32FC3); // pyramidCloud type 32FC3
        }
    }
    else
    {
        std::vector<Mat> pyramidCameraMatrix;
        buildPyramidCameraMatrix(cameraMatrix, (int)pyramidDepth.size(), pyramidCameraMatrix); // size: cloud=0 depth=4
        pyramidCloud.resize(pyramidDepth.size()); // size: cloud=4 depth=4
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
            CV_Assert(pyramidSobel[i].type() == CV_16SC1); // pyramidSobel type 16SC1
        }
    }
    else
    {
        pyramidSobel.resize(pyramidImage.size());
        
        for(size_t i = 0; i < pyramidImage.size(); i++)
        {
            Sobel(pyramidImage[i], pyramidSobel[i], CV_16S, dx, dy, sobelSize); // sobelSize=3
        }
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
            CV_Assert(pyramidTexturedMask[i].type() == CV_8UC1); // pyramidTexturedMask type 8UC1
        }
    }
    else
    {
        const float sobelScale2_inv = 1.f / (float)(sobelScale * sobelScale); //sobelscale=1/8, sobelScale2_inv=64
        pyramidTexturedMask.resize(pyramid_dI_dx.size());
        for(size_t i = 0; i < pyramidTexturedMask.size(); i++)
        {
            const float minScaledGradMagnitude2 = minGradMagnitudes[i] * minGradMagnitudes[i] * sobelScale2_inv; // 6400
            const Mat& dIdx = pyramid_dI_dx[i]; // // pyramid_dI_dx type 16SC1
            const Mat& dIdy = pyramid_dI_dy[i];

            Mat texturedMask(dIdx.size(), CV_8UC1, Scalar(0));
            for(int y = 0; y < dIdx.rows; y++)
            {
                
                const short *dIdx_row = dIdx.ptr<short>(y); // unsigned char 1 byte, unsigned short 2 Bytes, int 4 bytes, float 4 bytes, double 4 bytes
                const short *dIdy_row = dIdy.ptr<short>(y);
                uchar *texturedMask_row = texturedMask.ptr<uchar>(y);
                for(int x = 0; x < dIdx.cols; x++)
                {
                    // cout << dIdx_row[x] * dIdx_row[x] + dIdy_row[x] * dIdy_row[x] << " " << static_cast<float>(dIdx_row[x] * dIdx_row[x] + dIdy_row[x] * dIdy_row[x]) << " " << sizeof(static_cast<float>(dIdx_row[x] * dIdx_row[x] + dIdy_row[x] * dIdy_row[x])) << " " << sizeof(dIdx_row[x] * dIdx_row[x] + dIdy_row[x] * dIdy_row[x]) << endl;
                    float magnitude2 = static_cast<float>(dIdx_row[x] * dIdx_row[x] + dIdy_row[x] * dIdy_row[x]);
                    if(magnitude2 >= minScaledGradMagnitude2)
                        texturedMask_row[x] = 255;
                }
            }

            pyramidTexturedMask[i] = texturedMask & pyramidMask[i];
            randomSubsetOfMask(pyramidTexturedMask[i], (float)maxPointsPart); // maxPointsPart = 0.07 fix
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
    // cout << "Rt1" << Rt << R << endl;
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
                        // if(validMask0.at<uchar>(v0, u0)) // fix
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
    cout << "match-dir-points " << correspCount << endl;
}
// if(v1 < 40)
//     cout << v1 << "," << u1 << " " << d1 << " " << v0 << "," << u0 << " !!!!" << endl;
static inline
void calcRgbdEquationCoeffs(double* C, double dIdx, double dIdy, const Point3f& p3d, double fx, double fy)
{
    double invz  = 1. / p3d.z,
           v0 = dIdx * fx * invz,
           v1 = dIdy * fy * invz,
           v2 = -(v0 * p3d.x + v1 * p3d.y) * invz;
    // cout << dIdx << " " << fx << " " << invz << " " << v0 << endl;
    C[0] = -p3d.z * v1 + p3d.y * v2;
    C[1] =  p3d.z * v0 - p3d.x * v2;
    C[2] = -p3d.y * v0 + p3d.x * v1;
    C[3] = v0;
    C[4] = v1;
    C[5] = v2;
    // cout << C[0] << " " << C[1] << " " << C[2] << " " << C[3] << " " << C[4] << " " << C[5] << endl;
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
    // cout << "sigma: " << sigma << endl;
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
        //  cout << p0.x << " " << p0.y << " " << p0.z << endl;
         tp0.x = (float)(p0.x * Rt_ptr[0] + p0.y * Rt_ptr[1] + p0.z * Rt_ptr[2] + Rt_ptr[3]);
         tp0.y = (float)(p0.x * Rt_ptr[4] + p0.y * Rt_ptr[5] + p0.z * Rt_ptr[6] + Rt_ptr[7]);
         tp0.z = (float)(p0.x * Rt_ptr[8] + p0.y * Rt_ptr[9] + p0.z * Rt_ptr[10] + Rt_ptr[11]);

        //  func(A_ptr,
        //       w_sobelScale * dI_dx1.at<short int>(v1,u1),
        //       w_sobelScale * dI_dy1.at<short int>(v1,u1),
        //       tp0, fx, fy);
        // cout << dI_dx1.at<short int>(v1,u1) << " " << dI_dy1.at<short int>(v1,u1) << endl;
        func(A_ptr,
              dI_dx1.at<short int>(v1,u1),
              dI_dy1.at<short int>(v1,u1),
              tp0, fx, fy);
        for(int y = 0; y < transformDim; y++)
        {
            double* AtA_ptr = AtA.ptr<double>(y);
            for(int x = y; x < transformDim; x++){
                AtA_ptr[x] += A_ptr[y] * A_ptr[x];
                // cout << A_ptr[y]  << " " << A_ptr[x] << " " << A_ptr[y] * A_ptr[x] << endl;
            }
            // cout << "AtA_ptr " << AtA << endl;
            // AtB_ptr[y] += A_ptr[y] * w * diffs_ptr[correspIndex];
            AtB_ptr[y] += A_ptr[y] * diffs_ptr[correspIndex];
        }
        // cout << "AtA " << AtA << endl;
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

void calcPnPLsmMatrices(
    const Mat& K,
    const Mat& resultRt,
    Sophus::SE3d& pose,
    VecVector3d& pts_3d_eigen,
    VecVector2d& pts_2d_eigen,
    Eigen::Matrix<double, 6, 6>& H, 
    Vector6d& b,
    double& cost,
    double& lastCost,
    vector<double>& weight,
    Mat& AtA_pnp,
    Mat& AtB_pnp
){

    double fx = K.at<double>(0, 0);
    double fy = K.at<double>(1, 1);
    double cx = K.at<double>(0, 2);
    double cy = K.at<double>(1, 2);
    Eigen::Matrix<double, 4, 4> Rt;
    cv2eigen(resultRt, Rt);
    cout << "Rt " << Rt << endl;
    if(pts_3d_eigen.size() > 0){
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
            // cout <<  "[" << pts_2d_eigen[i][0] << ", " << pts_2d_eigen[i][1] << "]" << " [" << proj[0] << ", " << proj[1] << "]" << endl;
            // cout << fx << " " << inv_z << " " << inv_z2 << " " << fx << " " << fy << endl;
            cost += e.squaredNorm();
            Eigen::Matrix<double, 2, 6> J;
            J << fx * pc[0] * pc[1] * inv_z2,
                -fx - fx * pc[0] * pc[0] * inv_z2,
                fx * pc[1] * inv_z,
                -fx * inv_z,
                0,
                fx * pc[0] * inv_z2,
                fy + fy * pc[1] * pc[1] * inv_z2,
                -fy * pc[0] * pc[1] * inv_z2,
                -fy * pc[0] * inv_z,
                0,
                -fy * inv_z,
                fy * pc[1] * inv_z2;
                
            // cout << "J " << J << endl;
            // cout << endl;
            H += J.transpose() * (J * weight[i]);
            b += -J.transpose() * (e * weight[i]);
            // cout << J << endl;
            // cout << weight[i] << " " << e << endl;
        }
        cout << "cost: " << cost << " last cost: " << lastCost << endl;
    }

    
}
static
bool solveSystem(const Mat& AtA, const Mat& AtB, double detThreshold, Mat& x)
{
    double det = determinant(AtA);
    // cout << "solve system " << det << " " << AtA << endl;
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
    // cout << maxDepthDiff  << " " << maxTranslation << " " << maxRotation << " "  << method  << endl; // 0.07 0.15 15 3
    // for (int i=0; i<iterCounts.size(); i++) {
    //     cout << i << ":" << iterCounts[i] << endl;
    // }
    // cout << "RGBDICPOdometryImpl " << srcFrame->depth.type() << " " << dstFrame->depth.type() << endl;
    // iterCounts 7 7 7 10

    int transformDim = -1;
    CalcRgbdEquationCoeffsPtr rgbdEquationFuncPtr = 0;
    CalcICPEquationCoeffsPtr icpEquationFuncPtr = 0;
    switch(transfromType)
    {
    case Odometry::RIGID_BODY_MOTION:
        transformDim = 6;
        rgbdEquationFuncPtr = calcRgbdEquationCoeffs;
        icpEquationFuncPtr = calcICPEquationCoeffs;
        // cout << "rigid" << rgbdEquationFuncPtr << " " << icpEquationFuncPtr << calcRgbdEquationCoeffs << calcICPEquationCoeffs << endl; // 1 1
        break;
    case Odometry::ROTATION:
        transformDim = 3;
        rgbdEquationFuncPtr = calcRgbdEquationCoeffsRotation;
        icpEquationFuncPtr = calcICPEquationCoeffsRotation;
        cout << "rotation" << rgbdEquationFuncPtr << " " << icpEquationFuncPtr << endl;
        break;
    case Odometry::TRANSLATION:
        transformDim = 3;
        rgbdEquationFuncPtr = calcRgbdEquationCoeffsTranslation;
        icpEquationFuncPtr = calcICPEquationCoeffsTranslation;
        cout << "translation" << rgbdEquationFuncPtr << " " << icpEquationFuncPtr << endl;
        break;
    default:
        CV_Error(Error::StsBadArg, "Incorrect transformation type");
    }

    const int minOverdetermScale = 20;
    const int minCorrespsCount = minOverdetermScale * transformDim; //120
    const float icpWeight = 10.0;

    std::vector<Mat> pyramidCameraMatrix;
    
    buildPyramidCameraMatrix(cameraMatrix, (int)iterCounts.size(), pyramidCameraMatrix); // iterCounts.size() = 4
    
    Mat resultRt = initRt.empty() ? Mat::eye(4,4,CV_64FC1) : initRt.clone();
    Mat currRt, ksi;

    Mat resultRtPnP = initRt.empty() ? Mat::eye(4,4,CV_64FC1) : initRt.clone();
    Mat currRtPnP;
    //cout << "liyang test" << srcFrame->pyramidDepth[1] << endl;
    //cout << "liyang test" << srcFrame->pyramidMask[1] << endl;
    //exit(1);
    bool isOk = false;
    double lastCost = 0;
    Sophus::SE3d pose;
    for(int level = (int)iterCounts.size() - 1; level >= 0; level--) // 3 2 1 0
    {
        // cout << "level: " << level << " " << pyramidCameraMatrix[level] << endl;
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
        Mat AtA_pnp(cv::Size(6, 6), AtA_rgbd.type());
        Mat AtB_pnp(cv::Size(1, 6), AtB_rgbd.type());
        Mat corresps_rgbd, corresps_icp;

        vector<double> weight;
        Sophus::SE3d pose_gn;
        
        VecVector3d pts_3d_eigen;
        VecVector2d pts_2d_eigen;
        // Mat K = (Mat_<double>(3, 3) << 517.3f, 0, 318.6f, 0, 516.5f, 255.3f, 0, 0, 1);
        
        // Run transformation search on current level iteratively.
        
        for(int iter = 0; iter < iterCounts[level]; iter ++) // iter = 10 7 7 7
        {
            Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
            Vector6d b = Vector6d::Zero();
            double cost = 0;
            vector<double> weight;
            VecVector3d pts_3d_eigen;
            VecVector2d pts_2d_eigen;
            Mat resultRt_inv = resultRt.inv(DECOMP_SVD);
            if(method & RGBD_ODOMETRY){
                computeCorresps(levelCameraMatrix, levelCameraMatrix_inv, resultRt_inv,
                                srcLevelDepth, srcFrame->pyramidMask[level], dstLevelDepth, dstFrame->pyramidTexturedMask[level],
                                maxDepthDiff, corresps_rgbd);
            }

            if(method & ICP_ODOMETRY){
                computeCorresps(levelCameraMatrix, levelCameraMatrix_inv, resultRt_inv,
                                srcLevelDepth, srcFrame->pyramidMask[level], dstLevelDepth, dstFrame->pyramidNormalsMask[level],
                                maxDepthDiff, corresps_icp);
            }
            // cout << corresps_rgbd << " " << corresps_icp << endl; // size: oo*4
            // if(corresps_rgbd.rows < minCorrespsCount && corresps_icp.rows < minCorrespsCount && iter > 0 && cost >= lastCost){
            //     cout << "too few" << endl;
            //     break;
            // }
            if(level == 0){
                calculatePnPInliers(srcFrame->pyramidImage[level], dstFrame->pyramidImage[level], srcFrame->pyramidDepth[level], dstFrame->pyramidDepth[level], weight, pose, pts_3d_eigen, pts_2d_eigen, resultRt, levelCameraMatrix, 0);
            }
            // check residual error convergence
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
            if(corresps_icp.rows >= minCorrespsCount)
                calcPnPLsmMatrices(levelCameraMatrix, resultRt, pose, pts_3d_eigen, pts_2d_eigen, H, b, cost, lastCost, weight, AtA_pnp, AtB_pnp);
            
            cv::eigen2cv(H, AtA_pnp);
            cv::eigen2cv(b, AtB_pnp);
            AtA += AtA_pnp;
            AtB += AtB_pnp;

            bool solutionExist = solveSystem(AtA, AtB, determinantThreshold, ksi);
            // cout << "ksi " << ksi << endl;
            if(!solutionExist){ 
                break;
            }
            Eigen::Matrix<double, 6, 6> eigenH;
            Eigen::Matrix<double, 6, 1> eigenb;
            cv::cv2eigen(AtA, eigenH);
            cv::cv2eigen(AtB, eigenb);
            Vector6d dx = eigenH.ldlt().solve(eigenb);
            Vector6d new_dx;
            new_dx << dx[3], dx[4], dx[5], dx[0], dx[1], dx[2];
            // cout << "dx " << new_dx << endl;
            // cout << "old pose " << pose.matrix() << endl;
            pose = Sophus::SE3d::exp(new_dx) * pose;
            // cout << "new pose " << pose.matrix() << endl;
            
            if(transfromType == Odometry::ROTATION)
            {
                Mat tmp(6, 1, CV_64FC1, Scalar(0));
                ksi.copyTo(tmp.rowRange(0,3));
                ksi = tmp;
            }
            else if(transfromType == Odometry::TRANSLATION)
            {
                Mat tmp(6, 1, CV_64FC1, Scalar(0));
                ksi.copyTo(tmp.rowRange(3,6));
                ksi = tmp;
            }
            
            // cout << "ksi " << ksi << endl;
            // if(iter > 0 && cost >= lastCost){
            //     cout << "cost: " << cost << ", last cost: " << lastCost << endl;
            //     break;
            // }
            // lastCost = cost;
            // cout << "reverseRt " << currRtPnP << resultRtPnP << endl;
            // resultRtPnP = currRtPnP * resultRtPnP;

            computeProjectiveMatrix(ksi, currRt); // ksi size = 6*1
            // computeProjectiveMatrixInv(ksi, currRt);    
            // cout << "current" << currRt << resultRt << endl;
            resultRt = currRt * resultRt;
            // cout << "Rt " << resultRt << endl;
            isOk = true;
            // cout << "iteration " << iter << " cost=" << std::setprecision(12) << cost << endl;
            lastCost = cost;
            if (dx.norm() < 1e-8) { // converge
                cout << "converge" << endl;
                break;
            }
        
        }
        cout << "Finish calculating!!!!!" << endl;
    }

    Rt = resultRt;
    // cout << "Rt " << Rt << endl;
    cout << pose.matrix() << endl;
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
    // cout << "Rt by gn " << Rt << endl;
    if(isOk)
    {
        Mat deltaRt;

        if(initRt.empty())
            deltaRt = resultRt;
        else
            deltaRt = resultRt * initRt.inv(DECOMP_SVD);
        // cout << "initRt " << initRt << " " << deltaRt << endl;
        isOk = testDeltaTransformation(deltaRt, maxTranslation, maxRotation);
    }
    // cout << "isOk" << isOk << endl;
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
    cout << "there" << endl;
    return compute(srcFrame, dstFrame, Rt, initRt);
}

bool Odometry::compute(Ptr<OdometryFrame>& srcFrame, Ptr<OdometryFrame>& dstFrame, Mat& Rt, const Mat& initRt) const
{
    checkParams();
    Size srcSize = prepareFrameCache(srcFrame, OdometryFrame::CACHE_SRC);
    Size dstSize = prepareFrameCache(dstFrame, OdometryFrame::CACHE_DST);
    if(srcSize != dstSize)
        CV_Error(Error::StsBadSize, "srcFrame and dstFrame have to have the same size (resolution).");
    return computeImpl(srcFrame, dstFrame, Rt, initRt);
}

Size Odometry::prepareFrameCache(Ptr<OdometryFrame> &frame, int /*cacheType*/) const
{
    // cout << "1 " << Size() << endl;
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
                           minDepth(_minDepth), maxDepth(_maxDepth), maxDepthDiff(_maxDepthDiff), // mindepth fix
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
    // cout << "2" << endl;  // Rgbd is here
    cout << frame->image.empty() << " " << frame->depth.empty() << " " << frame->mask.empty() << endl;
    cout << frame->pyramidCloud.empty() << " " << frame->pyramidDepth.empty() << " " << frame->pyramidMask.empty() << endl;
    cout << frame->pyramidImage.empty() << " " << frame->pyramidNormals.empty() << " " << frame->pyramid_dI_dx.empty() << " " << frame->pyramid_dI_dy.empty() << " " << frame->pyramidTexturedMask.empty() << endl;
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
        }
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
    checkDepth(frame->depth, frame->image.size());
    // cout << "else          if " << frame->image.channels() << endl;
    // cout << "else          if " << frame->depth.channels() << endl;
    if(frame->mask.empty() && !frame->pyramidMask.empty())
        frame->mask = frame->pyramidMask[0];
    checkMask(frame->mask, frame->image.size());
    preparePyramidImage(frame->image, frame->pyramidImage, iterCounts.total()); // iterCounts size [1x4], total=4
    // cout << frame->pyramidImage[3].channels() << frame->pyramidImage[3].type() << endl;
    preparePyramidDepth(frame->depth, frame->pyramidDepth, iterCounts.total());
    // cout << frame->pyramidDepth[3].channels() << frame->pyramidDepth[3].type() << endl;

    preparePyramidMask(frame->mask, frame->pyramidDepth, (float)minDepth, (float)maxDepth,
                       frame->pyramidNormals, frame->pyramidMask);
    if(cacheType & OdometryFrame::CACHE_SRC)
        preparePyramidCloud(frame->pyramidDepth, cameraMatrix, frame->pyramidCloud);

    if(cacheType & OdometryFrame::CACHE_DST)
    {
        // for(int v1 = 0; v1 < frame->depth.rows; v1++)
        // {
        //     const float *depth1_row = frame->depth.ptr<float>(v1);
        //     for(int u1 = 0; u1 < frame->depth.cols; u1++)
        //     {
        //         float d1 = depth1_row[u1];
        //         cout << "origCache " << v1 << "," << u1 << " " << d1*5000 << endl;
        //     }
        // }
        // cout << "maxPoint " << maxPointsPart << endl; // maxPointsPart=0.7
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
    // minGradientMagnitudes.size() [1x4], iterCounts.size() [1x4]
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
    // cout << frame->pyramidMask.size() << " " << frame->mask.size() << " " << frame->mask.empty() << " " << frame->pyramidMask.empty() << endl;
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
