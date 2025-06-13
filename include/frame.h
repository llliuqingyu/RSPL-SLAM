#ifndef FRAME_H_
#define FRAME_H_

#include <string>
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <opencv2/opencv.hpp>

#include "utils.h"
#include "mappoint.h"
#include "mapline.h"
#include "camera.h"

#define FRAME_GRID_ROWS 48
#define FRAME_GRID_COLS 64

class Frame{
public:
  Frame();
  Frame(int frame_id, bool pose_fixed, CameraPtr camera, double timestamp);
  Frame& operator=(const Frame& other);

  void SetFrameId(int frame_id);
  int GetFrameId();
  double GetTimestamp();
  void SetPoseFixed(bool pose_fixed);
  bool PoseFixed();
  void SetPose(const Eigen::Matrix4d& pose);
  Eigen::Matrix4d& GetPose();

  // point features
  bool FindGrid(double& x, double& y, int& grid_x, int& grid_y);
  void AddFeatures(Eigen::Matrix<double, 259, Eigen::Dynamic>& features_left, 
      Eigen::Matrix<double, 259, Eigen::Dynamic>& features_right, std::vector<Eigen::Vector4d>& lines_left, 
      std::vector<Eigen::Vector4d>& lines_right, std::vector<cv::DMatch>& stereo_matches);
  void AddLeftFeatures(Eigen::Matrix<double, 259, Eigen::Dynamic>& features_left, std::vector<Eigen::Vector4d>& lines_left);
  int AddRightFeatures(Eigen::Matrix<double, 259, Eigen::Dynamic>& features_right, std::vector<Eigen::Vector4d>& lines_right, std::vector<cv::DMatch>& stereo_matches);

  Eigen::Matrix<double, 259, Eigen::Dynamic>& GetAllFeatures();

  size_t FeatureNum();

  bool GetKeypointPosition(size_t idx, Eigen::Vector3d& keypoint_pos);
  std::vector<cv::KeyPoint>& GetAllKeypoints();
  cv::KeyPoint& GetKeypoint(size_t idx);
  int GetInlierFlag(std::vector<bool>& inliers_feature_message);

  double GetRightPosition(size_t idx);
  std::vector<double>& GetAllRightPosition(); 

  bool GetDescriptor(size_t idx, Eigen::Matrix<double, 256, 1>& descriptor) const;

  double GetDepth(size_t idx);
  std::vector<double>& GetAllDepth();
  void SetDepth(size_t idx, double depth);

  void SetTrackIds(std::vector<int>& track_ids);
  std::vector<int>& GetAllTrackIds();
  void SetTrackId(size_t idx, int track_id);
  int GetTrackId(size_t idx);

  MappointPtr GetMappoint(size_t idx);
  std::vector<MappointPtr>& GetAllMappoints();
  void InsertMappoint(size_t idx, MappointPtr mappoint);

  bool BackProjectPoint(size_t idx, Eigen::Vector3d& p3D);
  CameraPtr GetCamera();
  void FindNeighborKeypoints(Eigen::Vector3d& p2D, std::vector<int>& indices, double r, bool filter = true) const;

  
  // line features
  size_t LineNum();
  void SetLineTrackId(size_t idx, int line_track_id);
  int GetLineTrackId(size_t idx);
  const std::vector<int>& GetAllLineTrackId();
  bool GetLine(size_t idx, Eigen::Vector4d& line);
  bool GetLineRight(size_t idx, Eigen::Vector4d& line);
  const std::vector<Eigen::Vector4d>& GatAllLines();
  // const std::vector<Eigen::Vector4d>& GatAllLinesbeforefilter();//    gai
  const std::vector<Eigen::Vector4d>& GatAllRightLines();
  bool GetRightLineStatus(size_t idx);
  const std::vector<bool>& GetAllRightLineStatus();
  void InsertMapline(size_t idx, MaplinePtr mapline);
  std::vector<MaplinePtr>& GetAllMaplines();
  const std::vector<MaplinePtr>& GetConstAllMaplines();
  std::map<int, double> GetPointsOnLine(size_t idx);
  const std::vector<std::map<int, double>>& GetPointsOnLines();
  bool TriangulateStereoLine(size_t idx, Vector6d& endpoints);
  void RemoveMapline(MaplinePtr mapline);
  void RemoveMapline(int idx);

  // covisibility graph
  void AddConnection(std::shared_ptr<Frame> frame, int weight);
  void AddConnection(std::set<std::pair<int, std::shared_ptr<Frame>>> connections);
  void SetParent(std::shared_ptr<Frame> parent);
  std::shared_ptr<Frame> GetParent();
  void SetChild(std::shared_ptr<Frame> child);
  std::shared_ptr<Frame> GetChild();

  void RemoveConnection(std::shared_ptr<Frame> frame);
  void RemoveMappoint(MappointPtr mappoint);
  void RemoveMappoint(int idx);
  void DecreaseWeight(std::shared_ptr<Frame> frame, int weight);

  std::vector<std::pair<int, std::shared_ptr<Frame>>> GetOrderedConnections(int number);

  void SetPreviousFrame(const std::shared_ptr<Frame> previous_frame);
  std::shared_ptr<Frame> PreviousFrame();
  
public:
  int tracking_frame_id;
  int local_map_optimization_frame_id;
  int local_map_optimization_fix_frame_id;

  // debug
  std::vector<int> line_left_to_right_match;
  std::vector<std::map<int, double>> relation_left;
  std::vector<std::map<int, double>> relation_right;

private:
  int _frame_id;//这里的frame_id是构造帧时被创建的，来自于rosmain时的递增设置
  double _timestamp;
  bool _pose_fixed;
  Eigen::Matrix4d _pose;//除了初始化，我没见计算pose啊？线特征三角化？

  // point features
  Eigen::Matrix<double, 259, Eigen::Dynamic> _features;// 这里存放的是左图的特征点
  std::vector<cv::KeyPoint> _keypoints;//frame对象的成员变量
  std::vector<int> _feature_grid[FRAME_GRID_COLS][FRAME_GRID_ROWS];//frame对象的成员变量，将特征点存放在格网中
  double _grid_width_inv;
  double _grid_height_inv;
  std::vector<double> _u_right;//保存左点对应的右点x像素值。恢复该特征点的深度用的
  std::vector<double> _depth;//三角化计算_depth深度信息，左图上的特征点
  std::vector<int> _track_ids;//当前帧的世界坐标系下的3D点，它不一定是从1开始的，连续着对应着所有特征点的一部分（自行体会），track_ids[i] = _track_id++;
  std::vector<MappointPtr> _mappoints;//保存当前帧的所有特征点

  // line features
  std::vector<Eigen::Vector4d> _lines;//也是左图的线特征
  std::vector<Eigen::Vector4d> _lines_right;//右图的线特征
  std::vector<bool> _lines_right_valid;//判断这条右图的线有没有和左图匹配上
  std::vector<std::map<int, double>> _points_on_lines;// 每帧的这个参数，vector大小跟line的个数一样，不管有没有点与该线匹配，都会push_back一个map，每个map中保存该点到直线的距离
  std::vector<int> _line_track_ids;
  std::vector<MaplinePtr> _maplines;

  // std::vector<Eigen::Vector4d> _lines_before_filter;//也是左图的线特征

  CameraPtr _camera;

  // covisibility graph
  std::map<std::shared_ptr<Frame>, int> _connections;
  std::set<std::pair<int, std::shared_ptr<Frame>>> _ordered_connections;
  std::shared_ptr<Frame> _parent;
  std::shared_ptr<Frame> _child;
  std::shared_ptr<Frame> _previous_frame;//上一帧
};

typedef std::shared_ptr<Frame> FramePtr;

#endif  // FRAME_H_