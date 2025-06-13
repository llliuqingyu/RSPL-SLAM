#ifndef MAP_BUILDER_H_
#define MAP_BUILDER_H_

#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>

#include "super_point.h"
#include "super_glue.h"
#include "read_configs.h"
#include "dataset.h"
#include "camera.h"
#include "frame.h"
#include "point_matching.h"
#include "line_processor.h"
#include "map.h"
#include "ros_publisher.h"
#include "g2o_optimization/types.h"

#include "rcf.h"

struct TrackingData{
  FramePtr frame;
  FramePtr ref_keyframe;//和谁进行的匹配
  std::vector<cv::DMatch> matches;
  InputDataPtr input_data;

  TrackingData() {}
  //深拷贝
  TrackingData& operator =(TrackingData& other){
		frame = other.frame;
		ref_keyframe = other.ref_keyframe;
		matches = other.matches;
		input_data = other.input_data;
		return *this;
	}
};
typedef std::shared_ptr<TrackingData> TrackingDataPtr;

class MapBuilder{
public:
  MapBuilder(Configs& configs);
  void AddInput(InputDataPtr data);
  void ExtractFeatureThread();
  void TrackingThread();

  void ExtractFeatrue(const cv::Mat& image, const cv::Mat& image_left_rcf, Eigen::Matrix<double, 259, Eigen::Dynamic>& points, std::vector<Eigen::Vector4d>& lines);// 形参：图片、(我添加一个跟图片保持一致的图片_rcf，用于提取图片的线特征)、特征点、特征线。返回这张图片的点线特征
  void ExtractFeatureAndMatch(const cv::Mat& image, const cv::Mat& image_right_rcf, const Eigen::Matrix<double, 259, Eigen::Dynamic>& points0, 
      Eigen::Matrix<double, 259, Eigen::Dynamic>& points1, std::vector<Eigen::Vector4d>& lines, std::vector<cv::DMatch>& matches);// 形参：图片A、(我添加一个跟图片A保持一致的图片A_rcf，用于提取图片A的线特征)、图片B的特征点、图片A的特征点、图片A的线特征、图片AB的特征点的匹配关系matches
  bool Init(FramePtr frame, cv::Mat& image_left, cv::Mat& image_right, cv::Mat& image_left_rcf, cv::Mat& image_right_rcf);
  int TrackFrame(FramePtr frame0, FramePtr frame1, std::vector<cv::DMatch>& matches);

  // pose_init = 0 : opencv pnp, pose_init = 1 : last frame pose, pose_init = 2 : original pose
  int FramePoseOptimization(FramePtr frame, std::vector<MappointPtr>& mappoints, std::vector<int>& inliers, int pose_init = 0);
  bool AddKeyframe(FramePtr last_keyframe, FramePtr current_frame, int num_match);
  void InsertKeyframe(FramePtr frame, const cv::Mat& image_right, const cv::Mat& image_right_rcf);
  void InsertKeyframe(FramePtr frame);

  // for tracking local map
  void UpdateReferenceFrame(FramePtr frame);
  void UpdateLocalKeyframes(FramePtr frame);
  void UpdateLocalMappoints(FramePtr frame);
  void SearchLocalPoints(FramePtr frame, std::vector<std::pair<int, MappointPtr>>& good_projections);
  int TrackLocalMap(FramePtr frame, int num_inlier_thr);

  void PublishFrame(FramePtr frame, cv::Mat& image);

  void SaveTrajectory();
  void SaveTrajectory(std::string file_path);
  void SaveMap(const std::string& map_root);

  void ShutDown();

private:
  // left feature extraction and tracking thread
  std::mutex _buffer_mutex;
  std::queue<InputDataPtr> _data_buffer;//先进先出
  std::thread _feature_thread;

  // pose estimation thread
  std::mutex _tracking_mutex;
  std::queue<TrackingDataPtr> _tracking_data_buffer;//有点意思，这里保存每一帧的结果：这一帧、这一帧的相关帧及上一个关键帧、左图与上一帧的匹配结果、这一帧的input结果
  std::thread _tracking_thread;

  // gpu mutex
  std::mutex _gpu_mutex;
  std::mutex _rcf_mutex;
  // std::thread _rcf_thread;

  bool _shutdown;

  // tmp 
  bool _init; // 初始为false
  int _track_id;//这里记录所有帧的特征点数，每个帧在计算Twc时，它都自增，将自增结果给每帧的track_id这样就能对应到每一帧中
  int _line_track_id;//这里记录所有线特征的个数
  FramePtr _last_frame; // 初始化了第一帧
  FramePtr _last_keyframe; // 初始化时，也被InsertKeyframe()函数将第一帧设置成关键帧
  int _num_since_last_keyframe;//距离上一个关键帧过了多久
  bool _last_frame_track_well;//这里应该是上一个关键帧跟踪的好坏

  cv::Mat _last_image; // image_left_rect
  cv::Mat _last_right_image; // image_right_rect
  cv::Mat _last_right_image_rcf; // image_right_rect_rcf

  cv::Mat _last_keyimage; // image_left_rect

  Pose3d _last_pose; 

  // for tracking local map
  bool _to_update_local_map;
  FramePtr _ref_keyframe; // 初始化了第一帧。该帧和谁进行的匹配
  std::vector<MappointPtr> _local_mappoints;
  std::vector<FramePtr> _local_keyframes;

  // class
  Configs _configs;//保存yaml配置的对象
  CameraPtr _camera;//保存相机配置yaml的对象
  SuperPointPtr _superpoint;//用于配置superpoint推理
  PointMatchingPtr _point_matching;//点匹配，内含superglue对象
  LineDetectorPtr _line_detector;
  RosPublisherPtr _ros_publisher;
  RCFPtr _rcf;

public:
  MapPtr _map;// 其中包含三角化的功能。所有最终的特征都存放在map中
};

#endif  // MAP_BUILDER_H_