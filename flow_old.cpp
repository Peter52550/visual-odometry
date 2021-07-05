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
using namespace std;
using namespace cv;

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
   
        
    for(int i = 1; !file.eof(); i++){
        string str;
        std::getline(file, str);
        datas[i] = str;
        if(str.empty()) break;
        if(str.at(0) == '#') continue; /* comment */
        cout << " previous image: " << datas[i-1] << "\n" << " current image "<< str << endl;
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
        
        // 建立3D点
        //Mat d1 = imread(depth1, IMREAD_UNCHANGED);       // 深度图为16位无符号数，单通道图像
        Mat K = (Mat_<double>(3, 3) << 517.3f, 0, 318.6f, 0, 516.5f, 255.3f, 0, 0, 1);
        vector<Point2f> pts_2d_old;
        vector<Point2f> pts_2d;
        int index = 1;
        for (DMatch m:matches) {
                ushort d = depth1.ptr<unsigned short>(int(keypoints_1[m.queryIdx].pt.y))[int(keypoints_1[m.queryIdx].pt.x)];
                // cout << "depth " << d << endl;
                if (d == 0){   // bad depth
                    continue;
                // d = 1;
                }
                // cout << "this is matches: " <<  index << endl;
                float dd = d / 5000.0;
                // cout << "keypoints_1[m.queryIdx].pt " << keypoints_1[m.queryIdx].pt << endl;
                Point2d p1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
                // cout << "p1.x " << p1.x << " p1.y " << p1.y << endl;
                // cout << "keypoints_2[m.trainIdx].pt " << keypoints_2[m.trainIdx].pt << endl;
                pts_2d_old.push_back(Point2f(p1.x * dd, p1.y * dd));
                pts_2d.push_back(keypoints_2[m.trainIdx].pt);
                index += 1;
            // }
        }
        bool b = false;
        // if(i ==373) {b = true;}
        cout << "2d-2d pairs: " << pts_2d_old.size() << " " << pts_2d.size() << " "  << i <<  endl;
        vector<Scalar> colors;
        cv::RNG rng;
        for(int i = 0; i < 100; i++){
            int r = rng.uniform(0, 256);
            int g = rng.uniform(0, 256);
            int b = rng.uniform(0, 256);
            colors.push_back(Scalar(r,g,b));
        }
        // vector<Point2f> p_old, p_new;
        // cv::goodFeaturesToTrack(image1, p_old, 100, 0.3, 7, Mat(), 7, false, 0.04);
        // // Calculate optical flow
        vector<uchar> status;
        vector<float> err;
        Mat mask = Mat::zeros(image1.size(), image1.type());
        cv::TermCriteria criteria = TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), 10, 0.03);
        cv::calcOpticalFlowPyrLK(image, image1, pts_2d_old, pts_2d, status, err, Size(15,15), 2, criteria);
        vector<Point2f> good_new;
        // Visualization part
        for(uint i = 0; i < pts_2d_old.size(); i++){
            // Select good points
            if(status[i] == 1) {
                good_new.push_back(pts_2d[i]);
                // Draw the tracks
                line(mask,pts_2d[i], pts_2d_old[i], colors[i], 2);
                circle(image, pts_2d[i], 5, colors[i], -1);
            }
        }
        // Display the demo
        Mat img;
        cv::add(image, mask, img);
        // if (save) {
        //     string save_path = "./optical_flow_frames/frame_" + to_string(counter) + ".jpg";
        //     imwrite(save_path, img);
        // }
        cv::imshow("flow", img);
        // int keyboard = cv::waitKey(25);
        // if (keyboard == 'q' || keyboard == 27)
        //     break;
        // Update the previous frame and previous points
        image1 = image.clone();
        pts_2d_old = good_new;

      
//       chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
//       Mat r, t, inliers;
//       solvePnPRansac(pts_3d, pts_2d, K, Mat(), r, t, b, 300, 0.05, 0.95, inliers);
//       // cout << inliers << inliers.size()  << inliers.at<int>(1,0)  << inliers.at<int>(3,0)<< endl;
//       for (int i=0; i<inliers.rows; i++){
//          cout << "inliers -> keypoints_1: " << keypoints_1[inliers.at<int>(i,0)].pt << "inliers -> keypoints_2: " << keypoints_2[inliers.at<int>(i,0)].pt << endl;
//       }
//       // cout << "inliers "<< inliers << endl;
//       cout<<"pnp OK = "<<b<<", inliers point num = "<<inliers.rows<<endl;
//       Mat R;
//       cv::Rodrigues(r, R); // r为旋转向量形式，用Rodrigues公式转换为矩阵
//       chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
//       chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
//       cout << "solve pnp in opencv cost time: " << time_used.count() << " seconds." << endl;

//       cout << "R=" << endl << R << endl;
//       cout << "t=" << endl << t << endl;
//       timestamps.push_back( timestap );
//       Mat output;
//       hconcat(R, t, output);
   
//       Mat Rt = Mat::eye(4,4,CV_64FC1);
//       Rt.at<double>(0,0) = R.at<double>(0,0);
//       Rt.at<double>(0,1) = R.at<double>(0,1);
//       Rt.at<double>(0,2) = R.at<double>(0,2);
//       Rt.at<double>(1,0) = R.at<double>(1,0);
//       Rt.at<double>(1,1) = R.at<double>(1,1);
//       Rt.at<double>(1,2) = R.at<double>(1,2);
//       Rt.at<double>(2,0) = R.at<double>(2,0);
//       Rt.at<double>(2,1) = R.at<double>(2,1);
//       Rt.at<double>(2,2) = R.at<double>(2,2);
//       Rt.at<double>(0,3) = t.at<double>(0,0);
//       Rt.at<double>(1,3) = t.at<double>(0,1);
//       Rt.at<double>(2,3) = t.at<double>(0,2);
//       cout << "Rt " << Rt << endl;  
//       Rts.push_back(Rt);

//       Mat Rt_ba;

//       VecVector3d pts_3d_eigen;
//       VecVector2d pts_2d_eigen;
//       for (size_t i = 0; i < pts_3d.size(); ++i) {
//          // cout << "vector3d " << Eigen::Vector3d(pts_3d[i].x, pts_3d[i].y, pts_3d[i].z) << "vector2d " << Eigen::Vector2d(pts_2d[i].x, pts_2d[i].y) << endl;
//          pts_3d_eigen.push_back(Eigen::Vector3d(pts_3d[i].x, pts_3d[i].y, pts_3d[i].z));
//          pts_2d_eigen.push_back(Eigen::Vector2d(pts_2d[i].x, pts_2d[i].y));
//       }
//       // for (size_t i = 0; i < pts_3d.size(); ++i) {
//       //    for (size_t j = 0; j < 3; ++j) {
//       //       cout << " eigen3d " << pts_3d_eigen[i][j] << cout << " eigen2d "  <<*pts_2d_eigen[i][j];
//       //    }
//       // }
//       cout << "calling bundle adjustment by g2o" << endl;
//       Sophus::SE3d pose_g2o;
//       t1 = chrono::steady_clock::now();
//       Rt_ba = bundleAdjustmentG2O(pts_3d_eigen, pts_2d_eigen, K, pose_g2o, argv[2], timestamps);
//       Rts_ba.push_back(Rt_ba);
//       t2 = chrono::steady_clock::now();
//       time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
//       cout << "solve pnp by g2o cost time: " << time_used.count() << " seconds." << endl;
   }
//    writeResults(argv[2], timestamps, Rts);
//    //writeResults(argv[2], timestamps, Rts_ba);
   
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

   vector<DMatch> match;
   matcher->match(descriptors_1, descriptors_2, match);
   double min_dist = 10000, max_dist = 0;
     for (int i = 0; i < descriptors_1.rows; i++) {
    double dist = match[i].distance;
     if (dist < min_dist) min_dist = dist;
     if (dist > max_dist) max_dist = dist;
   }
   printf("-- Max dist : %f \n", max_dist);
   printf("-- Min dist : %f \n", min_dist);
   for (int i = 0; i < descriptors_1.rows; i++) {
    if (match[i].distance <= max(2 * min_dist, 10.0)) {
       matches.push_back(match[i]);
     }
   }
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
   vector<DMatch> match;
   matcher->match(descriptors_1, descriptors_2, match);
   double min_dist = 10000, max_dist = 0;
     for (int i = 0; i < descriptors_1.rows; i++) {
    double dist = match[i].distance;
     if (dist < min_dist) min_dist = dist;
     if (dist > max_dist) max_dist = dist;
   }

   printf("-- Max dist : %f \n", max_dist);
   printf("-- Min dist : %f \n", min_dist);
   for (int i = 0; i < descriptors_1.rows; i++) {
    if (match[i].distance <= max(2 * min_dist, 10.0)) {
       matches.push_back(match[i]);
     }
   }
 }

 Point2d pixel2cam(const Point2d &p, const Mat &K) {
   return Point2d
     (
       (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
       (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
     );
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
  
//   writeResults(filename, timestamps, vertex_pose->estimate().matrix());
  pose = vertex_pose->estimate();
  return Rt;
}


