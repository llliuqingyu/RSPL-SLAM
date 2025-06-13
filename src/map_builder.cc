#include "map_builder.h"

#include <assert.h>
#include <iostream> 
#include <Eigen/Core> 
#include <Eigen/Geometry> 
#include <opencv2/core/eigen.hpp>

#include "super_point.h"
#include "super_glue.h"
#include "read_configs.h"
#include "dataset.h"
#include "camera.h"
#include "frame.h"
#include "point_matching.h"
#include "map.h"
#include "g2o_optimization/g2o_optimization.h"
#include "timer.h"
#include "debug.h"


MapBuilder::MapBuilder(Configs& configs): _shutdown(false), _init(false), _track_id(0), _line_track_id(0), 
    _to_update_local_map(false), _configs(configs){
  _camera = std::shared_ptr<Camera>(new Camera(configs.camera_config_path));
  _superpoint = std::shared_ptr<SuperPoint>(new SuperPoint(configs.superpoint_config));
  _rcf = std::shared_ptr<RCF>(new RCF());
  // if (!_superpoint->build()){
  //   std::cout << "Error in SuperPoint building" << std::endl;
  //   exit(0);
  // }
  if(_superpoint->build()){
      std::cout << "\033[36m SuperPoint model build successfully! \033[0m" << std::endl;
  }else{
      ROS_ERROR("SuperPoint engine deserialization failed.");    
      assert(false); // 这里应该是直接退出程序
  }

  if(_rcf->deserialize_engine()){
      std::cout << "\033[36m RCF model deserialized successfully! \033[0m" << std::endl;
  }else{
      assert(false); // 这里应该是直接退出程序
  }
  _point_matching = std::shared_ptr<PointMatching>(new PointMatching(configs.superglue_config));
  _line_detector = std::shared_ptr<LineDetector>(new LineDetector(configs.line_detector_config));
  _ros_publisher = std::shared_ptr<RosPublisher>(new RosPublisher(configs.ros_publisher_config));
  _map = std::shared_ptr<Map>(new Map(_configs.backend_optimization_config, _camera, _ros_publisher));

  _feature_thread = std::thread(boost::bind(&MapBuilder::ExtractFeatureThread, this));
  _tracking_thread = std::thread(boost::bind(&MapBuilder::TrackingThread, this));
}

//从ros话题中获取图片，并矫正它们，然后存在map对象的_data_buffer中。该函数在ros_main.cpp中使用，用于先把图片搞进map对象中。
void MapBuilder::AddInput(InputDataPtr data){
  cv::Mat image_left_rect, image_right_rect;
  // cv::Mat image_left_rect_rcf, image_right_rect_rcf;

  _camera->UndistortImage(data->image_left, data->image_right, image_left_rect, image_right_rect);
  // _camera->UndistortImage(data->image_left_rcf, data->image_right_rcf, image_left_rect_rcf, image_right_rect_rcf);

  data->image_left = image_left_rect;
  data->image_right = image_right_rect;

  while(_data_buffer.size() >= 3 && !_shutdown){ // 3,2000
    usleep(1000);
  }

  _buffer_mutex.lock();
  _data_buffer.push(data);
  _buffer_mutex.unlock();
}

void MapBuilder::ExtractFeatureThread(){
  while(!_shutdown){
    if(_data_buffer.empty()){
      usleep(1000);
      continue;
    }
    InputDataPtr input_data;
    _buffer_mutex.lock();
    input_data = _data_buffer.front();//按顺序拿到quene中的图片
    _data_buffer.pop();
    _buffer_mutex.unlock();


    int h = input_data->image_left.rows;
    int w = input_data->image_left.cols;
    cv::Mat image_left_rect2 = input_data->image_left.clone();
    cv::Mat image_right_rect2 = input_data->image_right.clone();
    cv::Mat image_left_rect_rcf2 = cv::Mat(h, w, CV_8UC1 );
    cv::Mat image_right_rect_rcf2 = cv::Mat(h, w, CV_8UC1 );
  
    // _camera->UndistortImage(input_data->image_left, input_data->image_right, image_left_rect2, image_right_rect2);
  
    // Infer RCF
    // cv::Mat image_left_rect_rcf, image_right_rect_rcf;
    std::vector<u_int> infer_vector_l, infer_vector_r;

    std::function<void()> rcf_infer = [&](){
      _rcf_mutex.lock();
      infer_vector_l = _rcf->infer(image_left_rect2);
      infer_vector_r = _rcf->infer(image_right_rect2);
      _rcf_mutex.unlock();
    };
    std::thread rcf_infer_thread(rcf_infer);
    rcf_infer_thread.join();

    if(infer_vector_l.size() != h * w || infer_vector_r.size() != h * w){
      std::cerr << "RCF inference output size mismatch: " << infer_vector_l.size() << " vs " << h * w << std::endl;
      continue;
    }
    for (int i = 0; i < h * w; ++i) {
      image_left_rect_rcf2.data[i] = static_cast<unsigned char>(infer_vector_l[i]);
    }
    for (int i = 0; i < h * w; ++i) {
      image_right_rect_rcf2.data[i] = static_cast<unsigned char>(infer_vector_r[i]);
    }

    // input_data->image_left_rcf = image_left_rect_rcf2.clone();
    // input_data->image_right_rcf = image_right_rect_rcf2.clone();
    input_data->image_left_rcf = image_left_rect_rcf2;
    input_data->image_right_rcf = image_right_rect_rcf2;
    std::cout << "\033[36m Image " << input_data->index << " rectified and rcf infered successfully! \033[0m" << std::endl;

    int frame_id = input_data->index;
    double timestamp = input_data->time;
    cv::Mat image_left_rect = input_data->image_left.clone();//为什么又要做一次复制呢？直接用input_data->image_left不是一样的吗
    cv::Mat image_right_rect = input_data->image_right.clone();
    cv::Mat image_left_rect_rcf = input_data->image_left_rcf.clone();// --------------改
    cv::Mat image_right_rect_rcf = input_data->image_right_rcf.clone();// --------------改

    // std::cout << "\033[31m ??????????????  \033[0m" << std::endl;

    // cv::imwrite("/home/lqy/airvo_for_github/src/rcf_infer_image/image_left_rect_rcf_" + std::to_string(frame_id) + ".jpg", image_left_rect_rcf);
    // cv::imwrite("/home/lqy/airvo_for_github/src/rcf_infer_image/image_left_rect_" + std::to_string(frame_id) + ".jpg", image_left_rect);

    // construct frame
    FramePtr frame = std::shared_ptr<Frame>(new Frame(frame_id, false, _camera, timestamp));//创建（开辟）一个帧对象，表示当前帧图片

    // std::cout << "\033[31m create img Problem????  \033[0m" << std::endl;

    // init
    if(!_init){
      // std::cout << "\033[31m Init????  \033[0m" << std::endl;
      _init = Init(frame, image_left_rect, image_right_rect, image_left_rect_rcf, image_right_rect_rcf);//返回ture
      _last_frame_track_well = _init;

      if(_init){
        _last_frame = frame;//这里写重复了
        _last_image = image_left_rect;//这没意义啊，这是函数的局部变量，函数结束就没了。cv::Mat内部有一个引用计数机制，指向同一块内存，类似于shared_ptr，牛逼
        _last_right_image = image_right_rect;
        _last_right_image_rcf = image_right_rect_rcf;
        _last_keyimage = image_left_rect;
      }
      PublishFrame(frame, image_left_rect);// 未初始化成功，也会publish这一帧
      continue;;
    }//完成初始化

    // extract features and track last keyframe
    FramePtr last_keyframe = _last_keyframe;//上一个关键帧
    const Eigen::Matrix<double, 259, Eigen::Dynamic> features_last_keyframe = last_keyframe->GetAllFeatures();

    std::vector<cv::DMatch> matches;
    Eigen::Matrix<double, 259, Eigen::Dynamic> features_left;//这里的left都是为了保存这一帧的数据。没右图啥事儿啊
    std::vector<Eigen::Vector4d> lines_left;
    ExtractFeatureAndMatch(image_left_rect, image_left_rect_rcf, features_last_keyframe, features_left, lines_left, matches);//当前帧与上一次关键帧之间对比，都是左图，相当与前后帧提取匹配
    frame->AddLeftFeatures(features_left, lines_left);//该帧的特征点和线特征，与关键帧的特征点和线特征匹配


    // 跟踪关系，相当于该帧与上一个关键帧之间的节点描述关系
    TrackingDataPtr tracking_data = std::shared_ptr<TrackingData>(new TrackingData());
    tracking_data->frame = frame;
    tracking_data->ref_keyframe = last_keyframe;// 相关帧也变成了上一个关键帧
    tracking_data->matches = matches;// 也是和上一个关键帧的匹配关系
    tracking_data->input_data = input_data;//相当于每一帧都要处理，且每一帧只与上一个关键帧做匹配
    
    while(_tracking_data_buffer.size() >= 2){// 为什么大于两帧还要睡眠一下
      usleep(1000);
    }

    // ROS_INFO("\033[33m Input track_data id is %d! \033[0m", frame->GetFrameId());
    _tracking_mutex.lock();
    _tracking_data_buffer.push(tracking_data);
    _tracking_mutex.unlock();
  }  
}

void MapBuilder::TrackingThread(){
  while(!_shutdown){
    if(_tracking_data_buffer.empty()){
      usleep(1000);
      continue;
    }

    TrackingDataPtr tracking_data;
    _tracking_mutex.lock();
    tracking_data = _tracking_data_buffer.front();
    _tracking_data_buffer.pop();
    _tracking_mutex.unlock();


    // std::cout << "\033[31m TrackingThread????  \033[0m" << std::endl;
    //怎么又掏出来一遍
    FramePtr frame = tracking_data->frame;
    FramePtr ref_keyframe = tracking_data->ref_keyframe;//保存的是上一个关键帧
    InputDataPtr input_data = tracking_data->input_data;
    std::vector<cv::DMatch> matches = tracking_data->matches;

    // ROS_INFO("\033[33m Get track_data id is %d! \033[0m", frame->GetFrameId());

    double timestamp = input_data->time;
    cv::Mat image_left_rect = input_data->image_left.clone();
    cv::Mat image_right_rect = input_data->image_right.clone();
    cv::Mat image_left_rect_rcf = input_data->image_left_rcf.clone();// --------------改
    cv::Mat image_right_rect_rcf = input_data->image_right_rcf.clone();// --------------改

    // track
    frame->SetPose(_last_frame->GetPose());//这里初始化后，拿到的是初始化的姿态。这里怎么直接赋值啊，不计算相邻帧吗
    std::function<int()> track_last_frame = [&](){
      if(_num_since_last_keyframe < 1 || !_last_frame_track_well) return -1;//如果上一个关键帧跟踪失败就卡死
      InsertKeyframe(_last_frame, _last_right_image, _last_right_image_rcf);//在提取特征的线程里，_last_frame已经被搞成当前帧了？
      _last_keyimage = _last_image;
      matches.clear();
      ref_keyframe = _last_frame;//相关帧要被重置成上一关键帧
      return TrackFrame(_last_frame, frame, matches);//matches都被清空了，还匹配个鬼啊
    };

    int num_match = matches.size();
    //这个if，如果匹配点较多，则认为是普通帧，经过TrackFrame函数得到当前帧的pose后就完了
    //如果匹配点较少，则认为是关键帧，进行track_last_frame函数
    if(num_match < _configs.keyframe_config.min_num_match){//如果匹配的点数小于最小匹配数，就认定为关键帧
      num_match = track_last_frame();
    }else{
      num_match = TrackFrame(ref_keyframe, frame, matches);//经过两次内点过滤，如果匹配点较少，则track_last_frame
      if(num_match < _configs.keyframe_config.min_num_match){
        num_match = track_last_frame();
      }
    }
    PublishFrame(frame, image_left_rect);//这里frame的pose已经被计算过了

    _last_frame_track_well = (num_match >= _configs.keyframe_config.min_num_match);
    if(!_last_frame_track_well) continue;

    frame->SetPreviousFrame(ref_keyframe);
    _last_frame_track_well = true;

    // for debug 
    // SaveTrackingResult(_last_keyimage, image_left, _last_keyframe, frame, matches, _configs.saving_dir);

    // 为什么这里又要做一次关键帧判断？
    if(AddKeyframe(ref_keyframe, frame, num_match) && ref_keyframe->GetFrameId() == _last_keyframe->GetFrameId()){
      InsertKeyframe(frame, image_right_rect, image_right_rect_rcf);
      _last_keyimage = image_left_rect;
    }

    ROS_INFO("\033[33m Frame id is %d , point_num_match: %d ! \033[0m", frame->GetFrameId(), num_match);

    _last_frame = frame;
    _last_image = image_left_rect;
    _last_right_image = image_right_rect;
    _last_right_image_rcf = image_right_rect_rcf; //------------改
  }  
}

/*
形参：图片、(我添加一个跟图片保持一致的图片_rcf，用于提取图片的线特征)、特征点、特征线。
作用：返回这张图片的点线特征

第一行：得分
第二行：x
第三行：y
剩下：描述子
*/
void MapBuilder::ExtractFeatrue(const cv::Mat& image, const cv::Mat& image_rcf, Eigen::Matrix<double, 259, Eigen::Dynamic>& points, 
    std::vector<Eigen::Vector4d>& lines){
  std::function<void()> extract_point = [&](){
    _gpu_mutex.lock();
    bool good_infer = _superpoint->infer(image, points);
    _gpu_mutex.unlock();
    if(!good_infer){
      std::cout << "Failed when extracting point features !" << std::endl;
      return;
    }
  };

  std::function<void()> extract_line = [&](){
    _line_detector->LineExtractor(image_rcf, lines);
  };

  std::thread point_ectraction_thread(extract_point);
  std::thread line_ectraction_thread(extract_line);

  point_ectraction_thread.join();
  line_ectraction_thread.join();
}

/*
形参：图片A、(我添加一个跟图片A保持一致的图片A_rcf，用于提取图片A的线特征)、图片B的特征点、图片A的特征点、图片A的线特征、图片AB的特征点的匹配关系matches

1、先superpoint推理出右图的特征点
2、_point_matching->MatchingPoints实则是superglue推理出匹配关系并保存到stereo_matches中
3、提取右图特征线
*/
void MapBuilder::ExtractFeatureAndMatch(const cv::Mat& image, const cv::Mat& image_rcf, const Eigen::Matrix<double, 259, Eigen::Dynamic>& points0, 
    Eigen::Matrix<double, 259, Eigen::Dynamic>& points1, std::vector<Eigen::Vector4d>& lines, std::vector<cv::DMatch>& matches){
  std::function<void()> extract_point_and_match = [&](){
    auto point0 = std::chrono::steady_clock::now();
     _gpu_mutex.lock();
    if(!_superpoint->infer(image, points1)){
      _gpu_mutex.unlock();
      std::cout << "Failed when extracting point features !" << std::endl;
      return;
    }
    auto point1 = std::chrono::steady_clock::now();

    matches.clear();
    _point_matching->MatchingPoints(points0, points1, matches);
    _gpu_mutex.unlock();
    auto point2 = std::chrono::steady_clock::now();
    auto point_time = std::chrono::duration_cast<std::chrono::milliseconds>(point1 - point0).count();
    auto point_match_time = std::chrono::duration_cast<std::chrono::milliseconds>(point2 - point1).count();
    // std::cout << "One Frame point Time: " << point_time << " ms." << std::endl;
    // std::cout << "One Frame point match Time: " << point_match_time << " ms." << std::endl;
  };

  std::function<void()> extract_line = [&](){
    auto line1 = std::chrono::steady_clock::now();
    _line_detector->LineExtractor(image_rcf, lines);
    auto line2 = std::chrono::steady_clock::now();
    auto line_time = std::chrono::duration_cast<std::chrono::milliseconds>(line2 - line1).count();
    // std::cout << "One Frame line Time: " << line_time << " ms." << std::endl;
  };

  auto feature1 = std::chrono::steady_clock::now();
  std::thread point_ectraction_thread(extract_point_and_match);
  std::thread line_ectraction_thread(extract_line);

  point_ectraction_thread.join();
  line_ectraction_thread.join();

  auto feature2 = std::chrono::steady_clock::now();
  auto feature_time = std::chrono::duration_cast<std::chrono::milliseconds>(feature2 - feature1).count();
  // std::cout << "One Frame featrue Time: " << feature_time << " ms." << std::endl;
}

/*
形参：当前帧FramePtr frame、左图、右图的clone栈对象

1、ExtractFeatrue提取左图的点、线特征（如果特征点个数小于150则return false）
2、ExtractFeatureAndMatch提取提取右图的点、线特征，并将左右点特征通过superglue匹配起来，存放在stereo_matches中
3、AddLeftFeatures
3、将左图的点线特征add到frame（上个函数的局部变量）中
*/
bool MapBuilder::Init(FramePtr frame, cv::Mat& image_left, cv::Mat& image_right, cv::Mat& image_left_rcf, cv::Mat& image_right_rcf){
  // extract features
  Eigen::Matrix<double, 259, Eigen::Dynamic> features_left, features_right;
  std::vector<Eigen::Vector4d> lines_left, lines_right;
  std::vector<cv::DMatch> stereo_matches;
  ExtractFeatrue(image_left, image_left_rcf, features_left, lines_left);// --------------------------改形参
  int feature_num = features_left.cols();
  if(feature_num < 150) return false;
  ExtractFeatureAndMatch(image_right, image_right_rcf, features_left, features_right, lines_right, stereo_matches);
  frame->AddLeftFeatures(features_left, lines_left);
  int stereo_point_match = frame->AddRightFeatures(features_right, lines_right, stereo_matches);
  if(stereo_point_match < 100) return false;

  // Eigen::Matrix4d init_pose = Eigen::Matrix4d::Identity();
  Eigen::Matrix4d init_pose;
  init_pose << 1,  0,  0,  0, 
               0,  0,  1,  0, 
               0, -1,  0,  1, 
               0,  0,  0,  1;
  // init_pose << 1, 0, 0, 0, 
  //              0, 1, 0, 0, 
  //              0, 0, 1, 0, 
  //              0, 0, 0, 1;
  frame->SetPose(init_pose);
  frame->SetPoseFixed(true);

  Eigen::Matrix3d Rwc = init_pose.block<3, 3>(0, 0);//从原矩阵的第 0 行第 0 列开始提取，提取的子块是一个 3x3 的矩阵
  Eigen::Vector3d twc = init_pose.block<3, 1>(0, 3);



  // construct mappoints            --三角化点特征并按照数据结构存在frame中
  int stereo_point_num = 0;
  std::vector<int> track_ids(feature_num, -1);//这里是左图所有的点，而不只是匹配点
  int frame_id = frame->GetFrameId();//得到第几帧
  Eigen::Vector3d tmp_position;
  std::vector<MappointPtr> new_mappoints;
  for(size_t i = 0; i < feature_num; i++){
    if(frame->BackProjectPoint(i, tmp_position)){//恢复该特征点的3D相机坐标。为什么要用这个所有点集。直接用stereo_point_match不就好了吗
      tmp_position = Rwc * tmp_position + twc;//左乘，将点从相机坐标系转（通过第一帧的T）到世界坐标系
      stereo_point_num++;
      track_ids[i] = _track_id++;
      Eigen::Matrix<double, 256, 1> descriptor;
      if(!frame->GetDescriptor(i, descriptor)) continue;// 从_features的第i列中获取描述子
      MappointPtr mappoint = std::shared_ptr<Mappoint>(new Mappoint(track_ids[i], tmp_position, descriptor));//这就固定好了一个世界坐标下的3D特征点
      mappoint->AddObverser(frame_id, i);//将帧id和这个点id绑定（我要知道这个点被哪几个帧看到了）
      frame->InsertMappoint(i, mappoint);// 将点也放在frame中一次
      new_mappoints.push_back(mappoint);//循环完成后直接将new_mappoints赋值给frame的_mappoints不是一样吗，都是std::vector<MappointPtr>类型-----这变量不是多余的吗
    }
  }
  frame->SetTrackIds(track_ids);//这是一个vector<int>，表示每个特征点的track_id，-1表示特征点没有被使用
  if(stereo_point_num < 100) return false;//100个点是不是太多了


  // construct maplines 和点保存是一个逻辑，注意线特征的Vector6d endpoints表示            --三角化点特征并按照数据结构存在frame中
  size_t line_num = frame->LineNum();
  std::vector<MaplinePtr> new_maplines;
  for(size_t i = 0; i < line_num; i++){
    frame->SetLineTrackId(i, _line_track_id);//为什么这里就是所有线了呢
    MaplinePtr mapline = std::shared_ptr<Mapline>(new Mapline(_line_track_id));
    Vector6d endpoints;
    if(frame->TriangulateStereoLine(i, endpoints)){//世界坐标系下的线段的两个端点坐标，应该还是左线                    里面TriangulateByStereo函数有点奇怪？？？？？？？？？？
      mapline->SetEndpoints(endpoints);//endpoints，将线段变成pu。。坐标系下，并保存到mapline自己的成员变量中
      mapline->SetObverserEndpointStatus(frame_id, 1);
    }else{
      mapline->SetObverserEndpointStatus(frame_id, 0);// 但这个程序默认好像使用的else情况
    }
    mapline->AddObverser(frame_id, i);//该帧的第i条线被frame_id观察到了，存放在mapline的_obverser中
    frame->InsertMapline(i, mapline);//将这条mapline放在frame中
    new_maplines.push_back(mapline);
    _line_track_id++;
  }



  // add frame and mappoints to map                  --处理帧的逻辑关系
  InsertKeyframe(frame);//第一帧当然是关键帧

  // 第一帧的mappoints和maplines都要存到map中，因为这是初始化，第一帧绝对可信
  for(MappointPtr mappoint : new_mappoints){//把所有点线都存到mapbuild的成员变量中了，那岂不是会很大吗？那我还要帧干嘛，帧和地图岂不是冲突了
    _map->InsertMappoint(mappoint);
  }
  for(MaplinePtr mapline : new_maplines){
    _map->InsertMapline(mapline);
  }
  _ref_keyframe = frame;
  _last_frame = frame;
  return true;
}

/*
形参：上一个关键帧、当前帧、两者的匹配关系
在这里才对上一关键帧和当前帧的特征线进行匹配
返回：
*/
int MapBuilder::TrackFrame(FramePtr frame0, FramePtr frame1, std::vector<cv::DMatch>& matches){
  // line tracking
  Eigen::Matrix<double, 259, Eigen::Dynamic>& features0 = frame0->GetAllFeatures();
  Eigen::Matrix<double, 259, Eigen::Dynamic>& features1 = frame1->GetAllFeatures();
  std::vector<std::map<int, double>> points_on_lines0 = frame0->GetPointsOnLines();
  std::vector<std::map<int, double>> points_on_lines1 = frame1->GetPointsOnLines();
  std::vector<int> line_matches;
  MatchLines(points_on_lines0, points_on_lines1, matches, features0.cols(), features1.cols(), line_matches);//在这里才进行线匹配

  std::vector<int> inliers(frame1->FeatureNum(), -1);//初始化inliers为-1，表示无效匹配
  std::vector<MappointPtr> matched_mappoints(features1.cols(), nullptr);//初始化当前帧中的点特征为nullptr，表示没有被计算到世界坐标下的3D坐标
  std::vector<MappointPtr>& frame0_mappoints = frame0->GetAllMappoints();//拿到关键帧的所有点特征（有效无效都有）
  for(auto& match : matches){
    int idx0 = match.queryIdx;
    int idx1 = match.trainIdx;
    matched_mappoints[idx1] = frame0_mappoints[idx0];//这相当于直接把上一帧与当前帧匹配的点特征的mappoints信息直接赋值给当前帧的mappoints
    //为什么要这么做呢？因为上一帧的mappoints是经过优化的，而当前帧的mappoints还没有经过优化，所以直接赋值可以减少计算量？
    inliers[idx1] = frame0->GetTrackId(idx0);
  }

  int num_inliers = FramePoseOptimization(frame1, matched_mappoints, inliers);//inliers是当前帧的特征点在上一帧中的track_id索引，-1表示无效匹配

  // update track id，因为上一个函数更新了inliers
  int RM = 0;
  if(num_inliers > _configs.keyframe_config.min_num_match){
    for(std::vector<cv::DMatch>::iterator it = matches.begin(); it != matches.end();){
      int idx0 = (*it).queryIdx;
      int idx1 = (*it).trainIdx;
      if(inliers[idx1] > 0){
        frame1->SetTrackId(idx1, frame0->GetTrackId(idx0));//相当于这一帧看到了上一帧的世界点，所以这一帧的track_id就等于上一帧的track_id
        frame1->InsertMappoint(idx1, frame0_mappoints[idx0]);//同上
      }

      if(inliers[idx1] > 0){
        it++;//迭代器向下移动一位
      }else{
        it = matches.erase(it);
        RM++;//移除了多少点
      }
    }
  }

  // update line track id 既然这个函数又过滤了一次内点，那为啥不在此处才进行线匹配呢
  const std::vector<MaplinePtr>& frame0_maplines = frame0->GetConstAllMaplines();//为啥获取的是上一帧的所有线特征？
  for(size_t i = 0; i < frame0_maplines.size(); i++){
    int j = line_matches[i];
    if(j < 0) continue;
    int line_track_id = frame0->GetLineTrackId(i);

    if(line_track_id >= 0){
      frame1->SetLineTrackId(j, line_track_id);//就这么处理线特征？？？？？？？？？
      frame1->InsertMapline(j, frame0_maplines[i]);
    }
  }

  return num_inliers;
}

// 形参：当前帧、当前帧的mappoints、inliers（当前帧的特征点在上一帧中的track_id索引，-1表示无效匹配）、pose_init（初始姿态）
// 作用：对当前帧的姿态进行优化（使用凸优化，并根据优化时，计算边的卡方值，重新过滤外点）
// 返回值：优化后的内点数量（如果num_inliers>_configs.keyframe_config.min_num_match，则给当前frame的pose赋值）
int MapBuilder::FramePoseOptimization(
    FramePtr frame, std::vector<MappointPtr>& mappoints, std::vector<int>& inliers, int pose_init){
  // solve PnP using opencv to get initial pose
  Eigen::Matrix4d Twc = Eigen::Matrix4d::Identity();
  std::vector<int> cv_inliers;
  //inliers是当前帧与上一帧匹配上的点匹配的索引中的内点（即有效点），但对应的是地图中有效的mappoints的id，是为了方便查找？
  int num_cv_inliers = SolvePnPWithCV(frame, mappoints, Twc, cv_inliers);//使用opencv的PnP算法来求解当前帧的初始姿态Twc

  Eigen::Vector3d check_dp = Twc.block<3, 1>(0, 3) - _last_frame->GetPose().block<3, 1>(0, 3);//检查两帧间隔距离
  if(check_dp.norm() > 0.5 || num_cv_inliers < _configs.keyframe_config.min_num_match){
    //这里应该是被认定为了关键帧。所以就用上一关键帧位姿做初始化？
    Twc = _last_frame->GetPose();
  }

  // Second, optimization
  MapOfPoses poses;
  MapOfPoints3d points;
  std::vector<CameraPtr> camera_list;
  VectorOfMonoPointConstraints mono_point_constraints;
  VectorOfStereoPointConstraints stereo_point_constraints;

  camera_list.emplace_back(_camera);// 搞这个干嘛，_camera不是常量吗

  // map of poses
  Pose3d pose;
  pose.p = Twc.block<3, 1>(0, 3);// PnPRansac求解的初始位姿
  pose.q = Twc.block<3, 3>(0, 0);
  int frame_id = frame->GetFrameId();
  poses.insert(std::pair<int, Pose3d>(frame_id, pose));//把当前帧的位姿插入到poses的map中

  // visual constraint construction  视觉约束构造
  std::vector<size_t> mono_indexes;
  std::vector<size_t> stereo_indexes;

  for(size_t i = 0; i < mappoints.size(); i++){
    // points
    MappointPtr mpt = mappoints[i];
    if(mpt == nullptr || !mpt->IsValid()) continue;
    Eigen::Vector3d keypoint; 
    if(!frame->GetKeypointPosition(i, keypoint)) continue;//这几步是为了，得到当前帧与上一帧匹配成功的点集（疑问？为什么不直接用ransac得到的点对）

    int mpt_id = mpt->GetId();//这个id在地图中是连续的，在帧上不连续
    Position3d point;
    point.p = mpt->GetPosition();
    point.fixed = true;//因为这是上一帧的地图点，所以固定
    points.insert(std::pair<int, Position3d>(mpt_id, point));

    // visual constraint
    if(keypoint(2) > 0){// 如果是立体点，那-1是啥？没印象
      //想起来了，这一帧与上一帧的匹配点，在这一帧里并不一定是立体点，可能是单目点，所以要判断一下
      StereoPointConstraintPtr stereo_constraint = std::shared_ptr<StereoPointConstraint>(new StereoPointConstraint()); 
      stereo_constraint->id_pose = frame_id;
      stereo_constraint->id_point = mpt_id;
      stereo_constraint->id_camera = 0;
      stereo_constraint->inlier = true;
      stereo_constraint->keypoint = keypoint;
      stereo_constraint->pixel_sigma = 0.8;
      stereo_point_constraints.push_back(stereo_constraint);
      stereo_indexes.push_back(i);
    }else{
      MonoPointConstraintPtr mono_constraint = std::shared_ptr<MonoPointConstraint>(new MonoPointConstraint()); 
      mono_constraint->id_pose = frame_id;
      mono_constraint->id_point = mpt_id;
      mono_constraint->id_camera = 0;
      mono_constraint->inlier = true;
      mono_constraint->keypoint = keypoint.head(2);
      mono_constraint->pixel_sigma = 0.8;
      mono_point_constraints.push_back(mono_constraint);
      mono_indexes.push_back(i);
    }

  }

  // 计算了poses, 并根据优化迭代计算边的卡方，重新计算了内点的数量
  int num_inliers = FrameOptimization(poses, points, camera_list, mono_point_constraints, 
      stereo_point_constraints, _configs.tracking_optimization_config);

  if(num_inliers > _configs.keyframe_config.min_num_match){
    // set frame pose
    Eigen::Matrix4d frame_pose = Eigen::Matrix4d::Identity();
    frame_pose.block<3, 3>(0, 0) = poses.begin()->second.q.matrix();
    frame_pose.block<3, 1>(0, 3) = poses.begin()->second.p;
    frame->SetPose(frame_pose);

    // update tracked mappoints把不是内点的点的track_id设置为-1
    for(size_t i = 0; i < mono_point_constraints.size(); i++){
      size_t idx = mono_indexes[i];
      if(!mono_point_constraints[i]->inlier){
        inliers[idx] = -1;
      }
    }

    for(size_t i = 0; i < stereo_point_constraints.size(); i++){
      size_t idx = stereo_indexes[i];
      if(!stereo_point_constraints[i]->inlier){
        inliers[idx] = -1;
      }
    }

  }

  return num_inliers;
}

// 形参：上一个关键帧、当前帧、匹配点数
// 作用：判断当前帧是否需要作为关键帧插入到地图中，判断条件：足够的匹配点？角度变化？位移变化？经过的帧数？
// 只要满足一个条件，就认为当前帧需要作为关键帧插入到地图中
bool MapBuilder::AddKeyframe(FramePtr last_keyframe, FramePtr current_frame, int num_match){
  Eigen::Matrix4d frame_pose = current_frame->GetPose();

  Eigen::Matrix4d& last_keyframe_pose = _last_keyframe->GetPose();
  Eigen::Matrix3d last_R = last_keyframe_pose.block<3, 3>(0, 0);
  Eigen::Vector3d last_t = last_keyframe_pose.block<3, 1>(0, 3);
  Eigen::Matrix3d current_R = frame_pose.block<3, 3>(0, 0);
  Eigen::Vector3d current_t = frame_pose.block<3, 1>(0, 3);

  Eigen::Matrix3d delta_R = last_R.transpose() * current_R;
  Eigen::AngleAxisd angle_axis(delta_R); 
  double delta_angle = angle_axis.angle();
  double delta_distance = (current_t - last_t).norm();
  int passed_frame_num = current_frame->GetFrameId() - _last_keyframe->GetFrameId();

  bool not_enough_match = (num_match < _configs.keyframe_config.max_num_match);
  bool large_delta_angle = (delta_angle > _configs.keyframe_config.max_angle);
  bool large_distance = (delta_distance > _configs.keyframe_config.max_distance);
  bool enough_passed_frame = (passed_frame_num > _configs.keyframe_config.max_num_passed_frame);
  return (not_enough_match || large_delta_angle || large_distance || enough_passed_frame);
}

// 形参：_last_frame, _last_right_image, _last_right_image_rcf
void MapBuilder::InsertKeyframe(FramePtr frame, const cv::Mat& image_right, const cv::Mat& image_right_rcf){
  _last_keyframe = frame;

  Eigen::Matrix<double, 259, Eigen::Dynamic> features_right;
  std::vector<Eigen::Vector4d> lines_right;
  std::vector<cv::DMatch> stereo_matches;

  // 对上一帧的左右图像进行特征提取和匹配？啊？上一帧？？？
  ExtractFeatureAndMatch(image_right, image_right_rcf, frame->GetAllFeatures(), features_right, lines_right, stereo_matches);
  frame->AddRightFeatures(features_right, lines_right, stereo_matches);
  InsertKeyframe(frame);
}

/*
形参：当前帧frame
更新关系  _last_keyframe = frame;
*/
void MapBuilder::InsertKeyframe(FramePtr frame){
  _last_keyframe = frame;

  // create new track id  分配该帧与mapbuilder中的，这不是在init做过了吗，这不相当于_track_id自增了两次吗？？？？
  std::vector<int>& track_ids = frame->GetAllTrackIds();
  for(size_t i = 0; i < track_ids.size(); i++){
    if(track_ids[i] < 0){// 什么意思？为什么没用的点也要设置？
      frame->SetTrackId(i, _track_id++);
    }
  }

  // create new line track id
  const std::vector<int>& line_track_ids = frame->GetAllLineTrackId();
  for(size_t i = 0; i < line_track_ids.size(); i++){
    if(line_track_ids[i] < 0){
      frame->SetLineTrackId(i, _line_track_id++);
    }
  }

  // insert keyframe to map
  _map->InsertKeyframe(frame);

  // update last keyframe
  _num_since_last_keyframe = 1;
  _ref_keyframe = frame;
  _to_update_local_map = true;
}

void MapBuilder::UpdateReferenceFrame(FramePtr frame){
  int current_frame_id = frame->GetFrameId();
  std::vector<MappointPtr>& mappoints = frame->GetAllMappoints();
  std::map<FramePtr, int> keyframes;
  for(MappointPtr mpt : mappoints){
    if(!mpt || mpt->IsBad()) continue;
    const std::map<int, int> obversers = mpt->GetAllObversers();
    for(auto& kv : obversers){
      int observer_id = kv.first;
      if(observer_id == current_frame_id) continue;
      FramePtr keyframe = _map->GetFramePtr(observer_id);
      if(!keyframe) continue;
      keyframes[keyframe]++;
    }
  }
  if(keyframes.empty()) return;

  std::pair<FramePtr, int> max_covi = std::pair<FramePtr, int>(nullptr, -1);
  for(auto& kv : keyframes){
    if(kv.second > max_covi.second){
      max_covi = kv;
    }
  }
 
  if(max_covi.first->GetFrameId() != _ref_keyframe->GetFrameId()){
    _ref_keyframe = max_covi.first;
    _to_update_local_map = true;
  }
}

void MapBuilder::UpdateLocalKeyframes(FramePtr frame){
  _local_keyframes.clear();
  std::vector<std::pair<int, FramePtr>> neighbor_frames = _ref_keyframe->GetOrderedConnections(-1);
  for(auto& kv : neighbor_frames){
    _local_keyframes.push_back(kv.second);
  }
}

void MapBuilder::UpdateLocalMappoints(FramePtr frame){
  _local_mappoints.clear();
  int current_frame_id = frame->GetFrameId();
  for(auto& kf : _local_keyframes){
    const std::vector<MappointPtr>& mpts = kf->GetAllMappoints();
    for(auto& mpt : mpts){
      if(mpt && mpt->IsValid() && mpt->tracking_frame_id != current_frame_id){
        mpt->tracking_frame_id = current_frame_id;
        _local_mappoints.push_back(mpt);
      }
    }
  }
}

void MapBuilder::SearchLocalPoints(FramePtr frame, std::vector<std::pair<int, MappointPtr>>& good_projections){
  int current_frame_id = frame->GetFrameId();
  std::vector<MappointPtr>& mpts = frame->GetAllMappoints();
  for(auto& mpt : mpts){
    if(mpt && !mpt->IsBad()) mpt->last_frame_seen = current_frame_id;
  }

  std::vector<MappointPtr> selected_mappoints;
  for(auto& mpt : _local_mappoints){
    if(mpt && mpt->IsValid() && mpt->last_frame_seen != current_frame_id){
      selected_mappoints.push_back(mpt);
    }
  }

  _map->SearchByProjection(frame, selected_mappoints, 1, good_projections);
}

int MapBuilder::TrackLocalMap(FramePtr frame, int num_inlier_thr){
  if(_to_update_local_map){
    UpdateLocalKeyframes(frame);
    UpdateLocalMappoints(frame);
  }

  std::vector<std::pair<int, MappointPtr>> good_projections;
  SearchLocalPoints(frame, good_projections);
  if(good_projections.size() < 3) return -1;

  std::vector<MappointPtr> mappoints = frame->GetAllMappoints();
  for(auto& good_projection : good_projections){
    int idx = good_projection.first;
    if(mappoints[idx] && !mappoints[idx]->IsBad()) continue;
    mappoints[idx] = good_projection.second;
  }

  std::vector<int> inliers(mappoints.size(), -1);
  int num_inliers = FramePoseOptimization(frame, mappoints, inliers, 2);

  // update track id
  if(num_inliers > _configs.keyframe_config.min_num_match && num_inliers > num_inlier_thr){
    for(size_t i = 0; i < mappoints.size(); i++){
      if(inliers[i] > 0){
        frame->SetTrackId(i, mappoints[i]->GetId());
        frame->InsertMappoint(i, mappoints[i]);
      }
    }
  }else{
    num_inliers = -1;
  }
  return num_inliers;
}

void MapBuilder::PublishFrame(FramePtr frame, cv::Mat& image){
  FeatureMessgaePtr feature_message = std::shared_ptr<FeatureMessgae>(new FeatureMessgae);
  FramePoseMessagePtr frame_pose_message = std::shared_ptr<FramePoseMessage>(new FramePoseMessage);

  feature_message->time = frame->GetTimestamp();
  feature_message->image = image;
  feature_message->keypoints = frame->GetAllKeypoints();;
  feature_message->lines = frame->GatAllLines();
  // feature_message->lines_before_filter = frame->GatAllLinesbeforefilter();
  feature_message->points_on_lines = frame->GetPointsOnLines();
  std::vector<bool> inliers_feature_message;
  frame->GetInlierFlag(inliers_feature_message);
  feature_message->inliers = inliers_feature_message;
  feature_message->line_track_ids = frame->GetAllLineTrackId();

  frame_pose_message->time = frame->GetTimestamp();
  frame_pose_message->pose = frame->GetPose();

  _ros_publisher->PublishFeature(feature_message);
  _ros_publisher->PublishFramePose(frame_pose_message);
}

void MapBuilder::SaveTrajectory(){
  std::string file_path = ConcatenateFolderAndFileName(_configs.saving_dir, "keyframe_trajectory.txt");
  _map->SaveKeyframeTrajectory(file_path);
}

void MapBuilder::SaveTrajectory(std::string file_path){
  _map->SaveKeyframeTrajectory(file_path);
}

void MapBuilder::SaveMap(const std::string& map_root){
  _map->SaveMap(map_root);
}

//合理的关闭两个子线程，防止数据出问题
void MapBuilder::ShutDown(){
  _shutdown = true;
  _feature_thread.join();//等待该线程执行完毕
  _tracking_thread.join();
}
