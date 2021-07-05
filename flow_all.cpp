#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/opencv.hpp>

#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <sophus/se3.hpp>
#include <Eigen/Core>


#include <chrono>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include<math.h>   
using namespace std;
using namespace cv;

#define UNKNOWN_FLOW_THRESH 1e9  


// bilinear interpolation
inline float GetPixelValue(const cv::Mat &img, float x, float y) {
    // boundary check
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x >= img.cols) x = img.cols - 1;
    if (y >= img.rows) y = img.rows - 1;
    uchar *data = &img.data[int(y) * img.step + int(x)];
    float xx = x - floor(x);
    float yy = y - floor(y);
    return float(
        (1 - xx) * (1 - yy) * data[0] +
        xx * (1 - yy) * data[1] +
        (1 - xx) * yy * data[img.step] +
        xx * yy * data[img.step + 1]
    );
}
//AVERAGE
double Average(vector<double> v)
{      double sum=0;
       for(int i=0;i<v.size();i++)
               sum+=v[i];
       return sum/v.size();
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
// Color encoding of flow vectors from:  
// http://members.shaw.ca/quadibloc/other/colint.htm  
// This code is modified from:  
// http://vision.middlebury.edu/flow/data/  
void makecolorwheel(vector<Scalar> &colorwheel)  
{  
    int RY = 15;  
    int YG = 6;  
    int GC = 4;  
    int CB = 11;  
    int BM = 13;  
    int MR = 6;  
  
    int i;  
  
    for (i = 0; i < RY; i++) colorwheel.push_back(Scalar(255,       255*i/RY,     0));  
    for (i = 0; i < YG; i++) colorwheel.push_back(Scalar(255-255*i/YG, 255,       0));  
    for (i = 0; i < GC; i++) colorwheel.push_back(Scalar(0,         255,      255*i/GC));  
    for (i = 0; i < CB; i++) colorwheel.push_back(Scalar(0,         255-255*i/CB, 255));  
    for (i = 0; i < BM; i++) colorwheel.push_back(Scalar(255*i/BM,      0,        255));  
    for (i = 0; i < MR; i++) colorwheel.push_back(Scalar(255,       0,        255-255*i/MR));  
}  
  
void motionToColor(Mat flow, Mat &color)  
{  
    if (color.empty())  
        color.create(flow.rows, flow.cols, CV_8UC3);  
  
    static vector<Scalar> colorwheel; //Scalar r,g,b  
    if (colorwheel.empty())  
        makecolorwheel(colorwheel);  
  
    // determine motion range:  
    float maxrad = -1;  
  
    // Find max flow to normalize fx and fy  
    for (int i= 0; i < flow.rows; ++i)   
    {  
        for (int j = 0; j < flow.cols; ++j)   
        {  
            Vec2f flow_at_point = flow.at<Vec2f>(i, j);  
            float fx = flow_at_point[0];  
            float fy = flow_at_point[1];  
            if ((fabs(fx) >  UNKNOWN_FLOW_THRESH) || (fabs(fy) >  UNKNOWN_FLOW_THRESH))  
                continue;  
            float rad = sqrt(fx * fx + fy * fy);  
            maxrad = maxrad > rad ? maxrad : rad;  
        }  
    }  
  
    for (int i= 0; i < flow.rows; ++i)   
    {  
        for (int j = 0; j < flow.cols; ++j)   
        {  
            uchar *data = color.data + color.step[0] * i + color.step[1] * j;  
            Vec2f flow_at_point = flow.at<Vec2f>(i, j);  
  
            float fx = flow_at_point[0] / maxrad;  
            float fy = flow_at_point[1] / maxrad;  
            if ((fabs(fx) >  UNKNOWN_FLOW_THRESH) || (fabs(fy) >  UNKNOWN_FLOW_THRESH))  
            {  
                data[0] = data[1] = data[2] = 0;  
                continue;  
            }  
            float rad = sqrt(fx * fx + fy * fy);  
  
            float angle = atan2(-fy, -fx) / CV_PI;  
            float fk = (angle + 1.0) / 2.0 * (colorwheel.size()-1);  
            int k0 = (int)fk;  
            int k1 = (k0 + 1) % colorwheel.size();  
            float f = fk - k0;  
            //f = 0; // uncomment to see original color wheel  
  
            for (int b = 0; b < 3; b++)   
            {  
                float col0 = colorwheel[k0][b] / 255.0;  
                float col1 = colorwheel[k1][b] / 255.0;  
                float col = (1 - f) * col0 + f * col1;  
                if (rad <= 1)  
                    col = 1 - rad * (1 - col); // increase saturation with radius  
                else  
                    col *= .75; // out of range  
                data[2 - b] = (int)(255.0 * col);  
            }  
        }  
    }  
}  
void writeResults( const string& filename, const vector<string>& timestamps, const vector<Mat>& Rt )
{
    CV_Assert( timestamps.size() == Rt.size() );

    ofstream file( filename.c_str() );
    if( !file.is_open() )
        return;

    cout.precision(4);
    for( size_t i = 0; i < Rt.size(); i++ )
    {
        const Mat& Rt_curr = Rt[i];
        if( Rt_curr.empty() )
            continue;

        CV_Assert( Rt_curr.type() == CV_64FC1 );

        Mat R = Rt_curr(Rect(0,0,3,3)), rvec;
        Rodrigues(R, rvec);
        double alpha = norm( rvec );
        if(alpha > DBL_MIN)
            rvec = rvec / alpha;

        double cos_alpha2 = std::cos(0.5 * alpha);
        double sin_alpha2 = std::sin(0.5 * alpha);

        rvec *= sin_alpha2;

        CV_Assert( rvec.type() == CV_64FC1 );
        // timestamp tx ty tz qx qy qz qw
        file << timestamps[i] << " " << fixed
             << Rt_curr.at<double>(0,3) << " " << Rt_curr.at<double>(1,3) << " " << Rt_curr.at<double>(2,3) << " "
             << rvec.at<double>(0) << " " << rvec.at<double>(1) << " " << rvec.at<double>(2) << " " << cos_alpha2 << endl;

    }
    file.close();
}

void find_feature_matches(
 const Mat &img_1, const Mat &img_2, std::vector<KeyPoint> &keypoints_1,vector<KeyPoint> &keypoints_2,std::vector<DMatch> &matches, const Mat &img_3);
 void find_feature_matches_another(
 const Mat &img_1, const Mat &img_2, std::vector<KeyPoint> &keypoints_1,vector<KeyPoint> &keypoints_2,std::vector<DMatch> &matches, const Mat &img_3);

// // 像素坐标转相机归一化坐标
 Point2d pixel2cam(const Point2d &p, const Mat &K);

// BA by g2o
typedef vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;
typedef vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> VecVector3d;
// BA by gauss-newton
Mat bundleAdjustmentGaussNewton(
  const VecVector3d &points_3d,
  const VecVector2d &points_2d,
  const VecVector3d &points_3d_nxt,
  const Mat &K,
  Sophus::SE3d &pose,
  int mode,
  const cv::Mat &img1,
  const cv::Mat &img2
);
double calc_residual(
   const VecVector3d &points_3d,
  const VecVector2d &points_2d,
  const VecVector3d &points_3d_nxt,
  Sophus::SE3d &pose,
  const Mat &K,
  vector<double>& residuals,
  const cv::Mat &img1,
  const cv::Mat &img2
);
void calc_residual_single(
   const VecVector3d &points_3d,
  const VecVector2d &points_2d,
  const VecVector3d &points_3d_nxt,
  Sophus::SE3d &pose,
  const Mat &K,
  vector<double>& residuals_pnp,
  vector<double>& residuals_icp,
  vector<double>& residuals_dir,
  vector<double>& res_std,
  const cv::Mat &img1,
  const cv::Mat &img2
);
Mat bundleAdjustmentG2O(
  const VecVector3d &points_3d,
  const VecVector2d &points_2d,
  const Mat &K,
  Sophus::SE3d &pose,
  const string& filename, 
  const vector<string>& timestamps
);

int main(int argc, char **argv) {
	if(argc != 4){
        cout << "Format: file_with_rgb_depth_pairs trajectory_file odometry_name [Rgbd or ICP or RgbdICP or FastICP]" << endl;
        return -1;
   }

   vector<string> timestamps;
   vector<Mat> Rts, Rts_ba;
    Rts.push_back(Mat::eye(4,4,CV_64FC1));
   Rts_ba.push_back(Mat::eye(4,4,CV_64FC1));
   const string filename = argv[1];
   ifstream file( filename.c_str() );
   if( !file.is_open() )
      return -1;
   char dlmrt = '/';
   size_t pos = filename.rfind(dlmrt);
   string dirname = pos == string::npos ? "" : filename.substr(0, pos) + dlmrt;

   const int timestampLength = 17;
   const int rgbPathLehgth = 17+8;
   const int depthPathLehgth = 17+10;

   float fx = 517.3f, // default
         fy = 516.5f,
         cx = 318.6f,
         cy = 255.3f;
   string datas[793];
   string str1;
   std::getline(file, str1);
    datas[0] = str1;
   string timestap3 = str1.substr(0, timestampLength);
   timestamps.push_back(timestap3);
        
    for(int i = 1; !file.eof(); i++){
        string str;
        std::getline(file, str);
        datas[i] = str;
        if(str.empty()) break;
        if(str.at(0) == '#') continue; /* comment */
        cout << " previous image: " << datas[i-1] << "\n" << " current image "<< str << endl;
        cout << "第 " << i << " 張圖片" << endl;
        Mat image, depth, image1, depth1, image2, depth2;
        // if(i > 2) {
        //     string rgbFilename2 = datas[i-2].substr(timestampLength + 1, rgbPathLehgth );
        //     string timestap2 = datas[i-2].substr(0, timestampLength);
        //     string depthFilename2 = datas[i-2].substr(2*timestampLength + rgbPathLehgth + 3, depthPathLehgth );

        //     image2 = imread(dirname + rgbFilename2);
        //     depth2 = imread(dirname + depthFilename2, -1);
        // }
        
        string rgbFilename1 = datas[i-1].substr(timestampLength + 1, rgbPathLehgth );
        string timestap1 = datas[i-1].substr(0, timestampLength);
        string depthFilename1 = datas[i-1].substr(2*timestampLength + rgbPathLehgth + 3, depthPathLehgth );
        image1 = imread(dirname + rgbFilename1);
        depth1 = imread(dirname + depthFilename1, -1);
    
        string rgbFilename = str.substr(timestampLength + 1, rgbPathLehgth );
        string timestap = str.substr(0, timestampLength);
        string depthFilename = str.substr(2*timestampLength + rgbPathLehgth + 3, depthPathLehgth );
        image = imread(dirname + rgbFilename);
        depth = imread(dirname + depthFilename, -1);
        // cout << "prev prev " << datas[i-2] << " previous image: " << datas[i-1] << " current image "<< str << endl;
        // CV_Assert(!image.empty());
        // CV_Assert(!depth.empty());
        // CV_Assert(!image1.empty());
        // CV_Assert(!depth1.empty());
        // CV_Assert(depth.type() == CV_16UC1);
        // CV_Assert(depth1.type() == CV_16UC1);

        // if(i > 2){
        //     CV_Assert(!image2.empty());
        //     CV_Assert(!depth2.empty()); 
        //     CV_Assert(depth2.type() == CV_16UC1);
        // }
        cout << CV_16UC1 << " " << image.type() << " " << image1.type() << " " << depth.type() << " " << depth1.type() << endl;
    cout << "hi " << image.channels() << " " << image1.channels() << endl;
        Mat prevgray, gray, flow, cflow;  
        // namedWindow("flow", WINDOW_NORMAL);  
        // Mat motion2color;   
        // chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
        // imshow("original", image1);  
        cvtColor(image1, prevgray, COLOR_BGR2GRAY);
        cvtColor(image, gray, COLOR_BGR2GRAY);
        cout << CV_16UC1 << " " << prevgray.type() << " " << gray.type() << " " << depth.type() << " " << depth1.type() << endl;
    cout << "hi " << prevgray.channels() << " " << gray.channels() << endl;
        // calcOpticalFlowFarneback(prevgray, gray, flow, 0.5, 3, 15, 3, 5, 1.2, 0); 
        // cout << flow.size() << endl;
        // motionToColor(flow, motion2color);  
        // imshow("flow", motion2color);
        
//         if(waitKey(10)>=0)  
//             break;  
        std::swap(image1, image);  
        // chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
        // chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
        // cout << "solve optical flow cost time: " << time_used.count() << " seconds." << endl;

        std::vector<KeyPoint> keypoints_1, keypoints_2, key1, key2;
        vector<DMatch> matches;
        Ptr<FeatureDetector> detector = ORB::create();
        detector->detect(image1, key1);
        detector->detect(image, key2);
        if(key1.size() == 0 || key2.size() == 0){
            find_feature_matches_another(image1, image, keypoints_1, keypoints_2, matches, image2);
            cout << "第二個: " <<  "一共找到了" << matches.size() << "组匹配点" << endl;
        }
        else{
            find_feature_matches(image1, image, keypoints_1, keypoints_2, matches, image2);
            cout << "第一個: " <<"一共找到了" << matches.size() << "组匹配点" << endl;
        }
        vector<Point2f> pt1, pt2;
        for (auto &kp: keypoints_1) pt1.push_back(kp.pt);
        vector<uchar> status;
        vector<float> error;
        chrono::steady_clock::time_point t3 = chrono::steady_clock::now();
        cv::calcOpticalFlowPyrLK(image1, image, pt1, pt2, status, error);
        chrono::steady_clock::time_point t4 = chrono::steady_clock::now();
        chrono::duration<double> time_used1 = chrono::duration_cast<chrono::duration<double>>(t4 - t3);
        cout << "optical flow by opencv: " << time_used1.count() << endl;
        // for(int i=0; i<pt1.size(); i++){
        //     cout << "pt1 " << pt1[i] << " pt2 " << pt2[i] << endl;
        // }
        cout << "pt1: " << pt1.size() << " pt2: " << pt2.size() << endl;
        // First, filter out the points with high error
        vector<Point2f> right_points_to_find;
        vector<int> right_points_to_find_back_index;
        for (unsigned int i=0; i<status.size(); i++) {
            if (status[i] && error[i] < 20.0) {
                // Keep the original index of the point in the optical flow array for future use
                right_points_to_find_back_index.push_back(i);
                // Keep the feature point itself
                right_points_to_find.push_back(pt2[i]);
            } else {
                status[i] = 0; // a bad flow
            }
        }
        // for each right_point see which detected feature it belongs to
        Mat right_points_to_find_flat = Mat(right_points_to_find).reshape(1,right_points_to_find.size()); //flatten array
        vector<Point2f> right_features; // detected features
        // KeyPointsToPoints(keypoints_2,right_features);
        // KeyPoint::convert(keypoints_2,right_features, 1, 1, 0, -1);
        std::vector<cv::KeyPoint>::iterator it;
        for( it= keypoints_2.begin(); it!= keypoints_2.end();it++){
            right_features.push_back(it->pt);
        }
        Mat right_features_flat = Mat(right_features).reshape(1,right_features.size());
        // Look around each OF point in the right image for any features that were detected in its area
        // and make a match.
        BFMatcher matcher(NORM_L2);
        // Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
        vector<vector<DMatch>> nearest_neighbors;
        matcher.radiusMatch(right_points_to_find_flat, right_features_flat, nearest_neighbors, 2.0f);
        // Check that the found neighbors are unique (throw away neighbors
        // that are too close together, as they may be confusing)
        std::set<int> found_in_right_points; // for duplicate prevention
        for(int i=0; i<nearest_neighbors.size(); i++) {
            DMatch _m;
            if(nearest_neighbors[i].size() == 1) {
            _m = nearest_neighbors[i][0]; // only one neighbor
            } else if(nearest_neighbors[i].size() > 1) {
                // 2 neighbors – check how close they are
                double ratio = nearest_neighbors[i][0].distance / nearest_neighbors[i][1].distance;
                if(ratio < 0.5) { // not too close //0.38
                // take the closest (first) one
                    _m = nearest_neighbors[i][0];
                }else { // too close – we cannot tell which is better
                    continue; // did not pass ratio test – throw away
                }
            } else {
                continue; // no neighbors
            }
            // prevent duplicates
            if (found_in_right_points.find(_m.trainIdx) == found_in_right_points.end()) {
                // The found neighbor was not yet used:
                // We should match it with the original indexing
                // ofthe left point
                _m.queryIdx = right_points_to_find_back_index[_m.queryIdx];
                matches.push_back(_m); // add this match
                found_in_right_points.insert(_m.trainIdx);
            }
        }
        cout<< "pruned " << matches.size() << " / " << nearest_neighbors.size() << " matches" << endl;
        // 建立3D点
        // Mat d1 = imread(depth1, IMREAD_UNCHANGED);       // 深度图为16位无符号数，单通道图像
        Mat K = (Mat_<double>(3, 3) << 517.3f, 0, 318.6f, 0, 516.5f, 255.3f, 0, 0, 1);
      vector<Point3f> pts_3d;
      vector<Point3f> pts_3d_nxt;
      vector<Point2f> pts_2d;
      std::vector<KeyPoint> keys1, keys2;
      std::vector<cv::Point2f> points1, points2;
      int index = 1;
      for (DMatch m:matches) {
         // if (index <keypoints_2.size()){
            ushort d = depth.ptr<unsigned short>(int(keypoints_1[m.queryIdx].pt.y))[int(keypoints_1[m.queryIdx].pt.x)];
            ushort d_nxt = depth1.ptr<unsigned short>(int(keypoints_2[m.trainIdx].pt.y))[int(keypoints_2[m.trainIdx].pt.x)];
            // cout << "depth " << d << endl;
            if (d == 0){   // bad depth
               continue;
            }
            if(d_nxt == 0){
                continue;
            }
            float d_new = d / 5000.0;
            float d_new_nxt = d / 5000.0;
            // cout << "keypoints_1[m.queryIdx].pt " << keypoints_1[m.queryIdx].pt << " keypoints_2[m.trainIdx].pt " << keypoints_2[m.trainIdx].pt << " depth " << d << endl;
            Point2d p1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
            Point2d p2 = pixel2cam(keypoints_2[m.trainIdx].pt, K);
            keys1.push_back(keypoints_1[m.queryIdx]);
            keys2.push_back(keypoints_2[m.trainIdx]);
            points1.push_back(keypoints_1[m.queryIdx].pt);
            points2.push_back(keypoints_2[m.trainIdx].pt);
            // cout << "p1.x " << p1.x << " p1.y " << p1.y << " depth " << dd << endl;
            
            pts_3d.push_back(Point3f(p1.x * d_new, p1.y * d_new, d_new));
            pts_3d_nxt.push_back(Point3f(p2.x * d_new_nxt, p2.y * d_new_nxt, d_new_nxt));
            pts_2d.push_back(keypoints_2[m.trainIdx].pt);
            index += 1;
         // }
      }
        bool b = false;
        // if(pts_3d.size() == 3) {b = true;}
        cout << "3d-2d pairs: " << pts_3d.size() << " " << pts_2d.size() << endl;
        keypoints_1.clear();
        keypoints_1 = keys1;
        keypoints_2.clear();
        keypoints_2 = keys2;
        
        // std::vector<cv::KeyPoint>::iterator it1, it2;
        // for( it1= keys1.begin(); it1!= keys1.end();it1++)
        // {
            
        // }
        // for( it2= keys2.begin(); it2!= keys2.end();it2++)
        // {
            
        // }
        // for (int i=0; i<; i++){
        //  cout << "inliers -> keypoints_1: " << keypoints_1[inliers.at<int>(i,0)].pt << " keypoints_2: " << keypoints_2[inliers.at<int>(i,0)].pt << endl;
        // }
        // for (int j=0; j<points1.size(); j++){
        //  cout << "pts1 " << points1[j] << " pts2: " << points2[j] << endl;
        // }
        vector<Scalar> colors;
        RNG rng;
        for(int j = 0; j < 100; j++){
            int r = rng.uniform(0, 256);
            int g = rng.uniform(0, 256);
            int b = rng.uniform(0, 256);
            colors.push_back(Scalar(r,g,b));
        }
        for(int j=0; j<keys2.size(); j++){
            circle(image1, points1[j], 5, colors[j], -1);
            circle(image, points2[j], 5, colors[j], -1);
        }
        if(i == 1){
            imwrite("0.jpg", image1);
            imwrite("1.jpg", image);
        }
        // // vector<Point2f> p_old, p_new;
        // // cv::goodFeaturesToTrack(image1, p_old, 100, 0.3, 7, Mat(), 7, false, 0.04);
        // // // Calculate optical flow
        // vector<uchar> status;
        // vector<float> err;
        // Mat mask = Mat::zeros(image1.size(), image1.type());
        // cv::TermCriteria criteria = TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), 10, 0.03);
        // cv::calcOpticalFlowPyrLK(image, image1, pts_2d_old, pts_2d, status, err, Size(15,15), 2, criteria);
        // vector<Point2f> good_new;
        // // Visualization part
        // for(uint i = 0; i < pts_2d_old.size(); i++){
        //     // Select good points
        //     if(status[i] == 1) {
        //         good_new.push_back(pts_2d[i]);
        //         // Draw the tracks
        //         line(mask,pts_2d[i], pts_2d_old[i], colors[i], 2);
        //         circle(image, pts_2d[i], 5, colors[i], -1);
        //     }
        // }
        // // Display the demo
        // Mat img;
        // cv::add(image, mask, img);
        // // if (save) {
        // //     string save_path = "./optical_flow_frames/frame_" + to_string(counter) + ".jpg";
        // //     imwrite(save_path, img);
        // // }
        // cv::imshow("flow", img);
        // // int keyboard = cv::waitKey(25);
        // // if (keyboard == 'q' || keyboard == 27)
        // //     break;
        // // Update the previous frame and previous points
        // image1 = image.clone();
        // pts_2d_old = good_new;

    
      chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
      Mat r, t, inliers;
      solvePnPRansac(pts_3d, pts_2d, K, Mat(), r, t, b, 1000, 6.0, 0.99, inliers, SOLVEPNP_ITERATIVE);
    //   cout << "inliers: " << inliers << endl;
      for (int i=0; i<inliers.rows; i++){
        //  cout << "inliers " << i << " -> keypoints_1: " << keypoints_1[inliers.at<int>(i,0)].pt << " keypoints_2: " << keypoints_2[inliers.at<int>(i,0)].pt << endl;
      }
      // cout << "inliers "<< inliers << endl;
      cout<<"pnp OK = "<< b <<", inliers point num = "<<inliers.rows<<endl;
      Mat R;
      cv::Rodrigues(r, R); // r为旋转向量形式，用Rodrigues公式转换为矩阵
      chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
      chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
      cout << "solve pnp in opencv cost time: " << time_used.count() << " seconds." << endl;

      cout << "R=" << endl << R << endl;
      cout << "t=" << endl << t << endl;
      timestamps.push_back( timestap );
      Mat output;
      hconcat(R, t, output);
   
      Mat Rt = Mat::eye(4,4,CV_64FC1);
      Rt.at<double>(0,0) = R.at<double>(0,0);
      Rt.at<double>(0,1) = R.at<double>(0,1);
      Rt.at<double>(0,2) = R.at<double>(0,2);
      Rt.at<double>(1,0) = R.at<double>(1,0);
      Rt.at<double>(1,1) = R.at<double>(1,1);
      Rt.at<double>(1,2) = R.at<double>(1,2);
      Rt.at<double>(2,0) = R.at<double>(2,0);
      Rt.at<double>(2,1) = R.at<double>(2,1);
      Rt.at<double>(2,2) = R.at<double>(2,2);
      Rt.at<double>(0,3) = t.at<double>(0,0);
      Rt.at<double>(1,3) = t.at<double>(0,1);
      Rt.at<double>(2,3) = t.at<double>(0,2);
      // cout << "Rt " << Rt << endl; 
      Mat& prevRt = *Rts.rbegin();
      // cout << "prevRt " << prevRt << endl;
      // cout << "Rt " << Rt << endl; 
    //   for (int i=0; i<inliers.rows; i++){
    //      Mat m( 4,1, CV_64FC1);
    //      m.at<double>(0,0) = keypoints_1[inliers.at<int>(i,0)].pt.x;
    //      m.at<double>(1,0) = keypoints_1[inliers.at<int>(i,0)].pt.y;
    //      m.at<double>(2,0) = pts_3d[i].z;
    //      m.at<double>(3,0) = 1;
    //      cout << "original " << m.t() << " projected " << (prevRt * Rt * m).t() << "  " << "actual " << keypoints_2[inliers.at<int>(i,0)].pt<< endl;
    //   }
      Rts.push_back(prevRt * Rt);
      

      Mat Rt_ba;
      VecVector3d pts_3d_eigen;
      VecVector3d pts_3d_eigen_nxt;
      VecVector2d pts_2d_eigen;
      for (size_t i = 0; i < pts_3d.size(); ++i) {
         // cout << "vector3d " << Eigen::Vector3d(pts_3d[i].x, pts_3d[i].y, pts_3d[i].z) << "vector2d " << Eigen::Vector2d(pts_2d[i].x, pts_2d[i].y) << endl;
         pts_3d_eigen.push_back(Eigen::Vector3d(pts_3d[i].x, pts_3d[i].y, pts_3d[i].z));
         pts_3d_eigen_nxt.push_back(Eigen::Vector3d(pts_3d_nxt[i].x, pts_3d_nxt[i].y, pts_3d_nxt[i].z));
         pts_2d_eigen.push_back(Eigen::Vector2d(pts_2d[i].x, pts_2d[i].y));
         
        //  pts_3d_eigen.push_back(Eigen::Vector3d(keypoints_1[inliers.at<int>(i,0)].pt.x, keypoints_1[inliers.at<int>(i,0)].pt.y, pts_3d[i].z));
        //  pts_2d_eigen.push_back(Eigen::Vector2d(keypoints_2[inliers.at<int>(i,0)].pt.x, keypoints_2[inliers.at<int>(i,0)].pt.y);
      }
      // for (size_t i = 0; i < pts_3d.size(); ++i) {
      //    for (size_t j = 0; j < 3; ++j) {
      //       cout << " eigen3d " << pts_3d_eigen[i][j] << cout << " eigen2d "  <<*pts_2d_eigen[i][j];
      //    }
      // }
      // cout << "calling bundle adjustment by g2o" << endl;
      // Sophus::SE3d pose_g2o;
      // t1 = chrono::steady_clock::now();
      // Rt_ba = bundleAdjustmentG2O(pts_3d_eigen, pts_2d_eigen, K, pose_g2o, argv[2], timestamps);
      // Mat& prevRtba = *Rts_ba.rbegin();
      // Rts_ba.push_back(prevRtba * Rt_ba);
      // t2 = chrono::steady_clock::now();
      // time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
      // cout << "solve pnp by g2o cost time: " << time_used.count() << " seconds." << endl;
    //  cout << "calling bundle adjustment by gauss newton" << endl;
     Sophus::SE3d pose_gn;
     t1 = chrono::steady_clock::now();
     int mode = 0; // 0=huber
     Mat Rt_baGauss = bundleAdjustmentGaussNewton(pts_3d_eigen, pts_2d_eigen,pts_3d_eigen_nxt, K, pose_gn, mode, image, image1);
      Mat& prevRtbaGauss = *Rts_ba.rbegin();
      cout << "prevRtBA " << prevRtbaGauss << endl;
      cout << "RtBA " << Rt_baGauss << endl; 
      
     t2 = chrono::steady_clock::now();
     time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    //  cout << "solve pnp by gauss newton cost time: " << time_used.count() << " seconds." << endl;
     for (int i=0; i<inliers.rows; i++){
         Mat m( 4,1, CV_64FC1);
         m.at<double>(0,0) = keypoints_1[inliers.at<int>(i,0)].pt.x;
         m.at<double>(1,0) = keypoints_1[inliers.at<int>(i,0)].pt.y;
         m.at<double>(2,0) = pts_3d[i].z;
         m.at<double>(3,0) = 1;
        //  cout << "original " << m.t() << " projected " << (prevRtbaGauss * Rt_baGauss * m).t() << "  " << "actual " << keypoints_2[inliers.at<int>(i,0)].pt<< endl;
      }
      Rts_ba.push_back(prevRtbaGauss * Rt_baGauss);
   }
//    writeResults(argv[2], timestamps, Rts);
   writeResults(argv[2], timestamps, Rts_ba);
   
  return 0;
}
void find_feature_matches_another(const Mat &img_1, const Mat &img_2,
                           std::vector<KeyPoint> &keypoints_1,
			                  std::vector<KeyPoint> &keypoints_2,
                           std::vector<DMatch> &matches,
                           const Mat &img_3) {
   Mat descriptors_1, descriptors_2, descriptors_3, descriptors_4;
   Ptr<FeatureDetector> detector = AgastFeatureDetector::create();
   Ptr<DescriptorExtractor> descriptor = AgastFeatureDetector::create();
   Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
   detector->detect(img_1, keypoints_1);
   detector->detect(img_2, keypoints_2);
   cout << "keypoints_1.size() "<< keypoints_1.size() << " keypoints_2.size() " << keypoints_2.size() << endl;
   // if(keypoints_1.size() != 0 && keypoints_2.size() != 0){
      for (int i=0; i< keypoints_1.size(); i++){
         // cout << "keypoint1 " << keypoints_1[i].pt << "keypoint2 " << keypoints_2[i].pt << endl;
      }
      descriptor->compute(img_1, keypoints_1, descriptors_1);
      descriptor->compute(img_2, keypoints_2, descriptors_2);
   // }
   int eee = descriptors_1.empty();
   int ddd = descriptors_2.empty();

//    vector<DMatch> match;
//    matcher->match(descriptors_1, descriptors_2, match);
//    double min_dist = 10000, max_dist = 0;
//      for (int i = 0; i < descriptors_1.rows; i++) {
//     double dist = match[i].distance;
//      if (dist < min_dist) min_dist = dist;
//      if (dist > max_dist) max_dist = dist;
//    }
//    printf("-- Max dist : %f \n", max_dist);
//    printf("-- Min dist : %f \n", min_dist);
//    for (int i = 0; i < descriptors_1.rows; i++) {
//     if (match[i].distance <= max(2 * min_dist, 10.0)) {
//        matches.push_back(match[i]);
//      }
//    }
 }
 void find_feature_matches(const Mat &img_1, const Mat &img_2,
                           std::vector<KeyPoint> &keypoints_1,
			                  std::vector<KeyPoint> &keypoints_2,
                           std::vector<DMatch> &matches,
                           const Mat &img_3) {

   Mat descriptors_1, descriptors_2, descriptors_3, descriptors_4;
   Ptr<FeatureDetector> detector = ORB::create();
   Ptr<DescriptorExtractor> descriptor = ORB::create();
   Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
   detector->detect(img_1, keypoints_1);
   detector->detect(img_2, keypoints_2);
   cout << "keypoints_1.size() "<< keypoints_1.size() << " keypoints_2.size() " << keypoints_2.size() << endl;
      for (int i=0; i< keypoints_1.size(); i++){
         // cout << "keypoint1 " << keypoints_1[i].pt << "keypoint2 " << keypoints_2[i].pt << endl;
      }
      descriptor->compute(img_1, keypoints_1, descriptors_1);
      descriptor->compute(img_2, keypoints_2, descriptors_2);
   int eee = descriptors_1.empty();
   int ddd = descriptors_2.empty();
//    vector<DMatch> match;
//    matcher->match(descriptors_1, descriptors_2, match);
//    double min_dist = 10000, max_dist = 0;
//      for (int i = 0; i < descriptors_1.rows; i++) {
//     double dist = match[i].distance;
//      if (dist < min_dist) min_dist = dist;
//      if (dist > max_dist) max_dist = dist;
//    }

//    printf("-- Max dist : %f \n", max_dist);
//    printf("-- Min dist : %f \n", min_dist);
//    for (int i = 0; i < descriptors_1.rows; i++) {
//     if (match[i].distance <= max(2 * min_dist, 10.0)) {
//        matches.push_back(match[i]);
//      }
//    }
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
  const VecVector3d &points_3d_nxt,
  Sophus::SE3d &pose,
  const Mat &K,
  vector<double>& residuals,
  const cv::Mat &img1,
  const cv::Mat &img2
){
    double fx = K.at<double>(0, 0);
  double fy = K.at<double>(1, 1);
  double cx = K.at<double>(0, 2);
  double cy = K.at<double>(1, 2);
  const int half_patch_size = 1;
  vector<double> residual_pnp;
  vector<double> residual_icp;
  vector<double> residual_dir;
  vector<double> res_std;
  for (int i=0; i<points_3d.size(); i++){
     Eigen::Vector3d pc = pose * points_3d[i];
     Eigen::Vector2d proj(fx * pc[0] / pc[2] + cx, fy * pc[1] / pc[2] + cy);
     Eigen::Vector2d error_pnp = points_2d[i] - proj;
     Eigen::Vector3d error_icp = points_3d_nxt[i] - pc;
     // direct with pose
     Eigen::Vector2d orig(fx * points_3d[i][0] / points_3d[i][2] + cx, fy * points_3d[i][1] / points_3d[i][2] + cy);
     double total_error;
     for (int x = -half_patch_size; x <= half_patch_size; x++){
        for (int y = -half_patch_size; y <= half_patch_size; y++) {
            double error_direct = GetPixelValue(img1, orig[0] + x, orig[1] + y) -
                            GetPixelValue(img2, proj[0] + x, proj[1] + y);
            total_error += error_direct;
        }
    }
    total_error /= 9;
    // cout << "total " << error_pnp.squaredNorm() << " " << error_icp.squaredNorm() << " " << total_error << endl;
    residual_pnp.push_back(error_pnp.squaredNorm());
    residual_icp.push_back(error_icp.squaredNorm());
    residual_dir.push_back(total_error);

    if (isnan(pc[2]) == false) {
         res_std.push_back(error_pnp.squaredNorm());
      }
    
  }
  double max_pnp = *max_element(residual_pnp.begin(), residual_pnp.end());
double min_pnp = *min_element(residual_pnp.begin(), residual_pnp.end());
double max_icp = *max_element(residual_icp.begin(), residual_icp.end());
double min_icp = *min_element(residual_icp.begin(), residual_icp.end());
double max_dir = *max_element(residual_dir.begin(), residual_dir.end());
double min_dir = *min_element(residual_dir.begin(), residual_dir.end());
double max_std = *max_element(res_std.begin(), res_std.end());
double min_std = *min_element(res_std.begin(), res_std.end());
for(int i=0; i<residual_pnp.size(); i++){
    double new_error_pnp = (residual_pnp[i] - min_pnp)/max_pnp;
    double new_error_icp = (residual_icp[i] - min_icp)/max_icp;
    double new_error_dir = (residual_dir[i] - min_dir)/max_dir;
    residuals.push_back(new_error_pnp + new_error_icp + new_error_dir);
    res_std[i] = (res_std[i] - min_std) / max_std;
    // cout << "error: " << new_error_pnp << " "  << new_error_icp << " " << new_error_dir << endl;
}
// residuals.push_back(error_pnp.squaredNorm());

  double avg = Average(res_std); 
  double std = Deviation(res_std,avg);
  return std;
}
void calc_residual_single(
   const VecVector3d &points_3d,
  const VecVector2d &points_2d,
  const VecVector3d &points_3d_nxt,
  Sophus::SE3d &pose,
  const Mat &K,
  vector<double>& residuals_pnp,
  vector<double>& residuals_icp,
  vector<double>& residuals_dir,
  vector<double>& res_std,
  const cv::Mat &img1,
  const cv::Mat &img2
){
    double fx = K.at<double>(0, 0);
  double fy = K.at<double>(1, 1);
  double cx = K.at<double>(0, 2);
  double cy = K.at<double>(1, 2);
  const int half_patch_size = 1;

  vector<double> res_std_pnp;
  vector<double> res_std_icp;
  vector<double> res_std_dir;
  for (int i=0; i<points_3d.size(); i++){
     Eigen::Vector3d pc = pose * points_3d[i];
     Eigen::Vector2d proj(fx * pc[0] / pc[2] + cx, fy * pc[1] / pc[2] + cy);
     Eigen::Vector2d error_pnp = points_2d[i] - proj;
     Eigen::Vector3d error_icp = points_3d_nxt[i] - pc;
     // direct with pose
     Eigen::Vector2d orig(fx * points_3d[i][0] / points_3d[i][2] + cx, fy * points_3d[i][1] / points_3d[i][2] + cy);
     double total_error;
     for (int x = -half_patch_size; x <= half_patch_size; x++){
        for (int y = -half_patch_size; y <= half_patch_size; y++) {
            double error_direct = GetPixelValue(img1, orig[0] + x, orig[1] + y) -
                            GetPixelValue(img2, proj[0] + x, proj[1] + y);
            total_error += error_direct;
        }
    }
    // cout << "total " << error_pnp.squaredNorm() << " " << error_icp.squaredNorm() << " " << total_error << endl;
    residuals_pnp.push_back(error_pnp.squaredNorm());
    residuals_icp.push_back(error_icp.squaredNorm());
    residuals_dir.push_back(total_error);

    if (isnan(pc[2]) == false) {
         res_std_pnp.push_back(error_pnp.squaredNorm());
         res_std_icp.push_back(error_icp.squaredNorm());
         res_std_dir.push_back(total_error);
    } 
  }
  double avg_pnp = Average(res_std_pnp); 
  double std_pnp = Deviation(res_std_pnp,avg_pnp);
  double avg_icp = Average(res_std_icp); 
  double std_icp = Deviation(res_std_icp,avg_icp);
  double avg_dir = Average(res_std_dir); 
  double std_dir = Deviation(res_std_dir,avg_dir);
  res_std.push_back(std_pnp);
  res_std.push_back(std_icp);
  res_std.push_back(std_dir);
}
Mat bundleAdjustmentGaussNewton(
  const VecVector3d &points_3d,
  const VecVector2d &points_2d,
  const VecVector3d &points_3d_nxt,
  const Mat &K,
  Sophus::SE3d &pose,
  int mode,
  const cv::Mat &img1,
  const cv::Mat &img2
  ) {
  typedef Eigen::Matrix<double, 6, 1> Vector6d;
  const int iterations = 100;
  const int half_patch_size = 1;
  double total_cost = 0, lastCost = 0, cost_pnp = 0, cost_icp = 0, cost_dir = 0, cost_dir_tmp = 0;
  double fx = K.at<double>(0, 0);
  double fy = K.at<double>(1, 1);
  double cx = K.at<double>(0, 2);
  double cy = K.at<double>(1, 2);
  int good_dir = 0;

   vector<double> residuals, residuals_pnp, residuals_icp, residuals_dir, res_std;
   
   // double res_std = calc_residual(points_3d, points_2d, points_3d_nxt, pose, K, residuals, img1, img2);
   calc_residual_single(points_3d, points_2d, points_3d_nxt, pose, K, residuals_pnp, residuals_icp, residuals_dir, res_std, img1, img2);
   // for(int i=0; i < residuals.size(); i++) {
   //    cout << residuals.at(i) << endl;
   // }
   // double huber_k = 1.345 * res_std * 2.5;
   // vector<double> weight;
   // if(mode == 0){
   //    for (int j=0; j<residuals.size(); j++){
   //       if(residuals[j] <= huber_k){
   //          weight.push_back(1.0);
   //       }else {
   //          weight.push_back(huber_k/residuals[j]);
   //       }
   //    }
   // }else {
   //    for (int j=0; j<residuals.size(); j++){
   //       weight.push_back(0.0);
   //    }
   // }
   double huber_k_pnp = 1.345 * res_std[0];
   double huber_k_icp = 1.345 * res_std[1];
   double huber_k_dir = 1.345 * res_std[2];
   vector<double> weight_pnp;
   vector<double> weight_icp;
   vector<double> weight_dir;
   if(mode == 0){
      for (int j=0; j<residuals_pnp.size(); j++){
         if(residuals_pnp[j] <= huber_k_pnp){
            weight_pnp.push_back(1.0);
         }else {
            weight_pnp.push_back(huber_k_pnp/residuals_pnp[j]);
         }
         if(residuals_icp[j] <= huber_k_icp){
            weight_icp.push_back(1.0);
         }else {
            weight_icp.push_back(huber_k_icp/residuals_icp[j]);
         }
         if(residuals_dir[j] <= huber_k_dir){
            weight_dir.push_back(1.0);
         }else {
            weight_dir.push_back(huber_k_dir/residuals_dir[j]);
         }
      }
   }else {
      for (int j=0; j<residuals_pnp.size(); j++){
         weight_pnp.push_back(0.0);
         weight_icp.push_back(0.0);
         weight_dir.push_back(0.0);
      }
   }
  for (int iter = 0; iter < iterations; iter++) {
    Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
    Vector6d b = Vector6d::Zero();
    Eigen::Matrix<double, 6, 6> H_pnp = Eigen::Matrix<double, 6, 6>::Zero();
    Vector6d b_pnp = Vector6d::Zero();
    Eigen::Matrix<double, 6, 6> H_icp = Eigen::Matrix<double, 6, 6>::Zero();
    Vector6d b_icp = Vector6d::Zero();
    Eigen::Matrix<double, 6, 6> H_dir = Eigen::Matrix<double, 6, 6>::Zero();
    Vector6d b_dir = Vector6d::Zero();
    

    total_cost = 0;
    cost_pnp = 0;
    cost_icp = 0;
    cost_dir = 0;
    cost_dir_tmp = 0;
    // compute cost
    for (int i = 0; i < points_3d.size(); i++) {
        
        Eigen::Vector3d pc = pose * points_3d[i];
        Eigen::Vector2d proj(fx * pc[0] / pc[2] + cx, fy * pc[1] / pc[2] + cy);
        Eigen::Vector2d orig(fx * points_3d[i][0] / points_3d[i][2] + cx, fy * points_3d[i][1] / points_3d[i][2] + cy);
    
        // cout << "T of F " << (proj[0] < half_patch_size || proj[0] > img2.cols - half_patch_size || proj[1] < half_patch_size ||
        //     proj[1] > img2.rows - half_patch_size) << endl;

        // cout << "nan " << isnan(pc[2]) << endl;

        if (proj[0] < half_patch_size || proj[0] > img2.cols - half_patch_size || proj[1] < half_patch_size ||
            proj[1] > img2.rows - half_patch_size)
            continue;
        if (isnan(pc[2]) == true) {
            continue;
        }
        // cout << "gooooooooooooooooooooooooood point" << endl;

         // PnP
        double inv_z = 1.0 / pc[2];
        double inv_z2 = inv_z * inv_z;
        Eigen::Vector2d error_pnp = points_2d[i] - proj;
        cost_pnp += error_pnp.squaredNorm();
        Eigen::Matrix<double, 2, 6> J_pnp;
        J_pnp << -fx * inv_z,
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

        // ICP
        Eigen::Vector3d error_icp = points_3d_nxt[i] - pc;
        Eigen::Matrix<double, 3, 6> J_icp;
        J_icp << 1, 0, 0, 0, pc[2], pc[1],
                0, 1, 0, -pc[2], 0, pc[0],
                0, 0, 1, pc[1], -pc[0], 0;
        cost_icp += error_icp.squaredNorm();


        // direct
        good_dir += 1;
        double total_error_dir = 0;
        double X = points_3d[i][0], Y = points_3d[i][1], Z = points_3d[i][2],
            Z2 = Z * Z, Z_inv = 1.0 / Z, Z2_inv = Z_inv * Z_inv;
        for (int x = -half_patch_size; x <= half_patch_size; x++){
            for (int y = -half_patch_size; y <= half_patch_size; y++) {

                double error_dir = GetPixelValue(img1, orig[0] + x, orig[1] + y) -
                            GetPixelValue(img2, proj[0] + x, proj[1] + y);
                Eigen::Matrix<double, 2, 6> J_pixel_xi;
                Eigen::Vector2d J_img_pixel;
                J_pixel_xi(0, 0) = fx * Z_inv;
                J_pixel_xi(0, 1) = 0;
                J_pixel_xi(0, 2) = -fx * X * Z2_inv;
                J_pixel_xi(0, 3) = -fx * X * Y * Z2_inv;
                J_pixel_xi(0, 4) = fx + fx * X * X * Z2_inv;
                J_pixel_xi(0, 5) = -fx * Y * Z_inv;

                J_pixel_xi(1, 0) = 0;
                J_pixel_xi(1, 1) = fy * Z_inv;
                J_pixel_xi(1, 2) = -fy * Y * Z2_inv;
                J_pixel_xi(1, 3) = -fy - fy * Y * Y * Z2_inv;
                J_pixel_xi(1, 4) = fy * X * Y * Z2_inv;
                J_pixel_xi(1, 5) = fy * X * Z_inv;

                J_img_pixel = Eigen::Vector2d(
                    0.5 * (GetPixelValue(img2, proj[0] + 1 + x, proj[1] + y) - GetPixelValue(img2, proj[0] - 1 + x, proj[1] + y)),
                    0.5 * (GetPixelValue(img2, proj[0] + x, proj[1] + 1 + y) - GetPixelValue(img2, proj[0] + x, proj[1] - 1 + y))
                );
                // total jacobian
                Vector6d J_dir = -1.0 * (J_img_pixel.transpose() * J_pixel_xi).transpose();
                cost_dir_tmp += error_dir * error_dir;

                H_dir += J_dir * J_dir.transpose()*weight_dir[i];
                b_dir += -error_dir * J_dir*weight_dir[i];
                total_error_dir += error_dir;
            }
        }
        cout << "errors in iteration " << iter << " " << error_pnp.squaredNorm() << " " << error_icp.squaredNorm() << " " << total_error_dir << endl;
        cout << "current weight " << weight_pnp[i] << " " << weight_icp[i] << " " << weight_dir[i]<< endl;


      //   if(error_icp.squaredNorm() < 1e-3 && fabs(total_error_dir) < 30){
      //       H_pnp += J_pnp.transpose() * (J_pnp * weight[i]);
      //       b_pnp += -J_pnp.transpose() * (error_pnp * weight[i]);

      //       H_icp += J_icp.transpose() * (J_icp * weight[i]);
      //       b_icp += -J_icp.transpose() * (error_icp * weight[i]);
      //   }else {
      //       H_pnp += J_pnp.transpose() * (J_pnp * weight[i]);
      //       b_pnp += -J_pnp.transpose() * (error_pnp * weight[i]);

      //       H_icp += J_icp.transpose() * (J_icp * weight[i]*0);
      //       b_icp += -J_icp.transpose() * (error_icp * weight[i]*0);
      //   }
      H_pnp += J_pnp.transpose() * (J_pnp * weight_pnp[i]);
      b_pnp += -J_pnp.transpose() * (error_pnp * weight_pnp[i]);

      H_icp += J_icp.transpose() * (J_icp * weight_icp[i]);
      b_icp += -J_icp.transpose() * (error_icp * weight_icp[i]);
    }
    cost_dir += cost_dir_tmp / good_dir;


    Vector6d dx;
   //  H = 0.8*H_pnp + 0.9*H_icp + 0.5*H_dir;
   //  b = 0.8*b_pnp + 0.9*b_icp + 0.5*b_dir;
    H = H_pnp + H_icp + H_dir;
    b = b_pnp + b_icp + b_dir;
    dx = H.ldlt().solve(b);

    
    if (isnan(dx[0])) {
      cout << "result is nan!" << endl;
      break;
    }

   //  total_cost = 0.8*cost_pnp + 0.9*cost_icp + 0.5*cost_dir;
    total_cost = cost_pnp + cost_icp + cost_dir;

    // if cost increase, break the loop
    if (iter > 0 && total_cost >= lastCost) {
      cout << "total cost: " << total_cost << ", last cost: " << lastCost << endl;
      break;
    }

    // update your estimation
    pose = Sophus::SE3d::exp(dx) * pose;
    
    lastCost = total_cost;

    cout << "iteration " << iter << " cost=" << std::setprecision(12) << total_cost << endl;
    if (dx.norm() < 1e-6) {
      // converge
      break;
    }
    
  }
   Mat Rt = Mat::eye(4,4,CV_64FC1);
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

    //   cout << "pose by g-n: \n" << pose.matrix() << endl;

      return Rt;
}

/// vertex and edges used in g2o ba
class VertexPose : public g2o::BaseVertex<6, Sophus::SE3d> {
   public:
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

   virtual void setToOriginImpl() override {
      _estimate = Sophus::SE3d();
   }

   /// left multiplication on SE3
   virtual void oplusImpl(const double *update) override {
      Eigen::Matrix<double, 6, 1> update_eigen;
      update_eigen << update[0], update[1], update[2], update[3], update[4], update[5];
      _estimate = Sophus::SE3d::exp(update_eigen) * _estimate;
   }

   virtual bool read(istream &in) override {}

   virtual bool write(ostream &out) const override {}
};

class EdgeProjection : public g2o::BaseUnaryEdge<2, Eigen::Vector2d, VertexPose> {
   public:
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

      EdgeProjection(const Eigen::Vector3d &pos, const Eigen::Matrix3d &K) : _pos3d(pos), _K(K) {}

   virtual void computeError() override {
      const VertexPose *v = static_cast<VertexPose *> (_vertices[0]);
      Sophus::SE3d T = v->estimate();
      Eigen::Vector3d pos_pixel = _K * (T * _pos3d);
      pos_pixel /= pos_pixel[2];
      _error = _measurement - pos_pixel.head<2>();
   }

   virtual void linearizeOplus() override {
      const VertexPose *v = static_cast<VertexPose *> (_vertices[0]);
      Sophus::SE3d T = v->estimate();
      Eigen::Vector3d pos_cam = T * _pos3d;
      double fx = _K(0, 0);
      double fy = _K(1, 1);
      double cx = _K(0, 2);
      double cy = _K(1, 2);
      double X = pos_cam[0];
      double Y = pos_cam[1];
      double Z = pos_cam[2];
      double Z2 = Z * Z;
      _jacobianOplusXi
         << -fx / Z, 0, fx * X / Z2, fx * X * Y / Z2, -fx - fx * X * X / Z2, fx * Y / Z,
         0, -fy / Z, fy * Y / (Z * Z), fy + fy * Y * Y / Z2, -fy * X * Y / Z2, -fy * X / Z;
   }

   virtual bool read(istream &in) override {}

   virtual bool write(ostream &out) const override {}

   private:
      Eigen::Vector3d _pos3d;
      Eigen::Matrix3d _K;
};

Mat bundleAdjustmentG2O(
  const VecVector3d &points_3d,
  const VecVector2d &points_2d,
  const Mat &K,
  Sophus::SE3d &pose,
  const string& filename, 
  const vector<string>& timestamps) {

  // 构建图优化，先设定g2o
  typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> BlockSolverType;  // pose is 6, landmark is 3
  typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType; // 线性求解器类型
  // 梯度下降方法，可以从GN, LM, DogLeg 中选
  auto solver = new g2o::OptimizationAlgorithmGaussNewton(
  g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
  g2o::SparseOptimizer optimizer;     // 图模型
  optimizer.setAlgorithm(solver);   // 设置求解器
  optimizer.setVerbose(true);       // 打开调试输出

  // vertex
  VertexPose *vertex_pose = new VertexPose(); // camera vertex_pose
  vertex_pose->setId(0);
  vertex_pose->setEstimate(Sophus::SE3d());
  optimizer.addVertex(vertex_pose);

  // K
  Eigen::Matrix3d K_eigen;
  K_eigen <<
          K.at<double>(0, 0), K.at<double>(0, 1), K.at<double>(0, 2),
    K.at<double>(1, 0), K.at<double>(1, 1), K.at<double>(1, 2),
    K.at<double>(2, 0), K.at<double>(2, 1), K.at<double>(2, 2);

  // edges
  int index = 1;
  for (size_t i = 0; i < points_2d.size(); ++i) {
    auto p2d = points_2d[i];
    auto p3d = points_3d[i];
    EdgeProjection *edge = new EdgeProjection(p3d, K_eigen);
    edge->setId(index);
    edge->setVertex(0, vertex_pose);
    edge->setMeasurement(p2d);
    edge->setInformation(Eigen::Matrix2d::Identity());
    optimizer.addEdge(edge);
    index++;
  }

  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
  optimizer.setVerbose(true);
  optimizer.initializeOptimization();
  optimizer.optimize(10);
  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "optimization costs time: " << time_used.count() << " seconds." << endl;
  cout << "pose estimated by g2o =\n" << vertex_pose->estimate().matrix() << endl;
//   cout << "pose matrix "  << vertex_pose->estimate().matrix()(0) << vertex_pose->estimate().matrix()(1) << vertex_pose->estimate().matrix()(4) << endl;

  Mat Rt = Mat::eye(4,4,CV_64FC1);
  Rt.at<double>(0,0) = vertex_pose->estimate().matrix()(0);
  Rt.at<double>(1,0) = vertex_pose->estimate().matrix()(1);
  Rt.at<double>(2,0) = vertex_pose->estimate().matrix()(2);
  Rt.at<double>(3,0) = vertex_pose->estimate().matrix()(3);
  Rt.at<double>(0,1) = vertex_pose->estimate().matrix()(4);
  Rt.at<double>(1,1) = vertex_pose->estimate().matrix()(5);
  Rt.at<double>(2,1) = vertex_pose->estimate().matrix()(6);
  Rt.at<double>(3,1) = vertex_pose->estimate().matrix()(7);
  Rt.at<double>(0,2) = vertex_pose->estimate().matrix()(8);
  Rt.at<double>(1,2) = vertex_pose->estimate().matrix()(9);
  Rt.at<double>(2,2) = vertex_pose->estimate().matrix()(10);
  Rt.at<double>(3,2) = vertex_pose->estimate().matrix()(11);
  Rt.at<double>(0,3) = vertex_pose->estimate().matrix()(12);
  Rt.at<double>(1,3) = vertex_pose->estimate().matrix()(13);
  Rt.at<double>(2,3) = vertex_pose->estimate().matrix()(14);
  Rt.at<double>(3,3) = vertex_pose->estimate().matrix()(15);

  pose = vertex_pose->estimate();
  return Rt;
}

