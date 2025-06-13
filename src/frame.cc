#include "frame.h"
#include <assert.h>

#include "line_processor.h"

Frame::Frame(){
}

Frame::Frame(int frame_id, bool pose_fixed, CameraPtr camera, double timestamp):
    tracking_frame_id(-1), local_map_optimization_frame_id(-1), local_map_optimization_fix_frame_id(-1),
    _frame_id(frame_id), _pose_fixed(pose_fixed), _camera(camera), _timestamp(timestamp){
  _grid_width_inv = static_cast<double>(FRAME_GRID_COLS)/static_cast<double>(_camera->ImageWidth());
  _grid_height_inv = static_cast<double>(FRAME_GRID_ROWS)/static_cast<double>(_camera->ImageHeight());
}

Frame& Frame::operator=(const Frame& other){
  tracking_frame_id = other.tracking_frame_id;
  local_map_optimization_frame_id = other.local_map_optimization_frame_id;
  local_map_optimization_fix_frame_id = other.local_map_optimization_fix_frame_id;
  _frame_id = other._frame_id;
  _timestamp = other._timestamp;
  _pose_fixed = other._pose_fixed;
  _pose = other._pose;

  _features = other._features;
  _keypoints = other._keypoints;
  for(int i=0;i<FRAME_GRID_COLS;i++){
    for(int j=0; j<FRAME_GRID_ROWS; j++){
      _feature_grid[i][j] = other._feature_grid[i][j];
    }
  }
  _grid_width_inv = other._grid_width_inv;
  _grid_height_inv = other._grid_height_inv;
  _u_right = other._u_right;
  _depth = other._depth;
  _track_ids = other._track_ids;
  _mappoints = other._mappoints;
  _camera = other._camera;
  return *this;
}

void Frame::SetFrameId(int frame_id){
  _frame_id = frame_id;
}

int Frame::GetFrameId(){
  return _frame_id;
}

double Frame::GetTimestamp(){
  return _timestamp;
}

void Frame::SetPoseFixed(bool pose_fixed){
  _pose_fixed = pose_fixed;
}

bool Frame::PoseFixed(){
  return _pose_fixed;
}

void Frame::SetPose(const Eigen::Matrix4d& pose){
  _pose = pose;
}

Eigen::Matrix4d& Frame::GetPose(){
  return _pose;
}

bool Frame::FindGrid(double& x, double& y, int& grid_x, int& grid_y){
  grid_x = std::round(x * _grid_width_inv);
  grid_y = std::round(y * _grid_height_inv);

  grid_x = std::min(std::max(0, grid_x), (FRAME_GRID_COLS-1));
  grid_y = std::min(std::max(0, grid_y), (FRAME_GRID_ROWS-1));

  return !(grid_x < 0 || grid_x >= FRAME_GRID_COLS || grid_y < 0 || grid_y >= FRAME_GRID_ROWS);
}

void Frame::AddFeatures(Eigen::Matrix<double, 259, Eigen::Dynamic>& features_left, 
    Eigen::Matrix<double, 259, Eigen::Dynamic>& features_right, std::vector<Eigen::Vector4d>& lines_left, 
    std::vector<Eigen::Vector4d>& lines_right, std::vector<cv::DMatch>& stereo_matches){
  AddLeftFeatures(features_left, lines_left);
  AddRightFeatures(features_right, lines_right, stereo_matches);
}

/*
形参：左图特征点、左图特征线
1、挑选特征点分数满足要求的点，并将它们存放在该frame的_keypoints中，并且还将它们都放进_feature_grid中（并未要求唯一性）
2、初始化_u_right和_depth的vector大小
3、初始化_track_ids和_mappoints的大小
4、通过AssignPointsToLines将点和线固定关系到_points_on_lines中
5、初始化_line_track_ids和_maplines。
这些初始化都是针对当前frame的初始化
*/
void Frame::AddLeftFeatures(Eigen::Matrix<double, 259, Eigen::Dynamic>& features_left, 
    std::vector<Eigen::Vector4d>& lines_left){
  _features = features_left;

  // fill in keypoints and assign features to grids
  size_t features_left_size = _features.cols();
  for(size_t i = 0; i < features_left_size; ++i){
    double score = _features(0, i);
    double x = _features(1, i);
    double y = _features(2, i);
    _keypoints.emplace_back(x, y, 8, -1, score);

    int grid_x, grid_y;
    bool found = FindGrid(x, y, grid_x, grid_y);
    assert(found);//assert 宏接受一个布尔表达式作为参数，如果该表达式为 true，则程序继续正常执行；如果表达式为 false，则程序会终止并报告一个断言失败错误
    _feature_grid[grid_x][grid_y].push_back(i);
  }

  // initialize u_right and depth
  _u_right = std::vector<double>(features_left_size, -1);
  _depth = std::vector<double>(features_left_size, -1);

  // initialize track_ids and mappoints
  std::vector<int> track_ids(features_left_size, -1); // 每个初始值都是-1
  SetTrackIds(track_ids);
  std::vector<MappointPtr> mappoints(features_left_size, nullptr);// 这些都是空指针，还没计算mappoint呢
  _mappoints = mappoints;

  // assign points to lines
  _lines = lines_left;
  std::vector<std::map<int, double>> points_on_line_left;
  std::vector<int> line_matches;
  AssignPointsToLines(lines_left, features_left, points_on_line_left);
  _points_on_lines = points_on_line_left;

  // initialize line track ids and maplines
  size_t line_num = lines_left.size();
  std::vector<int> line_track_ids(line_num, -1);
  _line_track_ids = line_track_ids;
  std::vector<MaplinePtr> maplines(line_num, nullptr);
  _maplines = maplines;

  // for debug
  relation_left = points_on_line_left;
}

/*
形参：右图点、右图线、stereo_matches
1、根据最小像素差来过滤一次匹配点stereo_matches
2、三角化计算_depth深度信息，_u_right[idx_left] = features_right(1, idx_right)保存左点对应的右点x像素值
3、通过AssignPointsToLines将点和线固定关系到_points_on_lines中
4、MatchLines函数进行左图右图的线特征匹配
返回matches.size()是根据视差过滤后的点匹配数量
*/
int Frame::AddRightFeatures(Eigen::Matrix<double, 259, Eigen::Dynamic>& features_right, 
    std::vector<Eigen::Vector4d>& lines_right, std::vector<cv::DMatch>& stereo_matches){
  // filter matches from superglue，第一次通过视差过滤一下匹配点
  std::vector<cv::DMatch> matches;
  double min_x_diff = _camera->MinXDiff();
  double max_x_diff = _camera->MaxXDiff();
  const double max_y_diff = _camera->MaxYDiff();
  for(cv::DMatch& match : stereo_matches){
    int idx_left = match.queryIdx;// 查询描述符的索引（即在第一个图像中的特征点索引）
    int idx_right = match.trainIdx;//训练描述符的索引（即在第二个图像中的特征点索引）

    double dx = std::abs(_features(1, idx_left) - features_right(1, idx_right));// 特征点x的差值
    double dy = std::abs(_features(2, idx_left) - features_right(2, idx_right));// 特征点y的差值

    if(dx > min_x_diff && dx < max_x_diff && dy <= max_y_diff){
      matches.emplace_back(match);
    }
  }

  // triangulate stereo points
  for(cv::DMatch& match : matches){
    int idx_left = match.queryIdx;
    int idx_right = match.trainIdx;

    assert(idx_left < _u_right.size());
    _u_right[idx_left] = features_right(1, idx_right);
    _depth[idx_left] = _camera->BF() / (_features(1, idx_left) - features_right(1, idx_right));
  }

  // assign points to lines
  std::vector<std::map<int, double>> points_on_line_right;
  AssignPointsToLines(lines_right, features_right, points_on_line_right);

  // match stereo lines
  std::vector<int> line_matches;
  size_t line_num = _lines.size();
  _lines_right.resize(line_num);// 初始化_lines_right的大小与左图
  _lines_right_valid.resize(line_num);
  MatchLines(_points_on_lines, points_on_line_right, matches, _features.cols(), features_right.cols(), line_matches);
  for(size_t i = 0; i < line_num; i++){
    if(line_matches[i] > 0){
      _lines_right[i] = lines_right[line_matches[i]];
      _lines_right_valid[i] = true;
    }else{
      _lines_right_valid[i] = false;
    }
  }

  // for debug
  line_left_to_right_match = line_matches;
  relation_right = points_on_line_right;

  return matches.size();
}

Eigen::Matrix<double, 259, Eigen::Dynamic>& Frame::GetAllFeatures(){
  return _features;
}

size_t Frame::FeatureNum(){
  return _features.cols();
}

// 返回 Vector3d，前两位是左图的xy，第三位是右图的x
bool Frame::GetKeypointPosition(size_t idx, Eigen::Vector3d& keypoint_pos){
  if(idx > _features.cols()) return false;// 特征点索引未超过
  keypoint_pos.head(2) = _features.block<2, 1>(1, idx);//获取向量的前 n 个元素，并返回一个新的向量
  keypoint_pos(2) = _u_right[idx];//得到
  return true;
}

std::vector<cv::KeyPoint>& Frame::GetAllKeypoints(){
  return _keypoints;
}

cv::KeyPoint& Frame::GetKeypoint(size_t idx){
  assert(idx < _keypoints.size());
  return _keypoints[idx];
}

int Frame::GetInlierFlag(std::vector<bool>& inliers_feature_message){
  int num_inliers = 0;
  inliers_feature_message.resize(_mappoints.size());
  for(size_t i = 0; i < _mappoints.size(); i++){
    if(_mappoints[i] && !_mappoints[i]->IsBad()){
      inliers_feature_message[i] = true;
      num_inliers++;
    }else{
      inliers_feature_message[i] = false;
    }
  }
  return num_inliers;
}

double Frame::GetRightPosition(size_t idx){
  assert(idx < _u_right.size());
  return _u_right[idx];
}

std::vector<double>& Frame::GetAllRightPosition(){
  return _u_right;
} 

// 设置点的描述子，并传递给descriptor。这函数
bool Frame::GetDescriptor(size_t idx, Eigen::Matrix<double, 256, 1>& descriptor) const{
  if(idx > _features.cols()) return false;
  descriptor = _features.block<256, 1>(3, idx);
  return true;
}

double Frame::GetDepth(size_t idx){
  assert(idx < _depth.size());
  return _depth[idx];
}

std::vector<double>& Frame::GetAllDepth(){
  return _depth;
}

void Frame::SetDepth(size_t idx, double depth){
  assert(idx < _depth.size());
  _depth[idx] = depth;
}

/*
将这一帧的trackid按照竹节形式，添加到这一帧的track_ids中
*/
void Frame::SetTrackIds(std::vector<int>& track_ids){
  _track_ids = track_ids;
}

std::vector<int>& Frame::GetAllTrackIds(){
  return _track_ids;
}

void Frame::SetTrackId(size_t idx, int track_id){
  _track_ids[idx] = track_id;
}

int Frame::GetTrackId(size_t idx){
  assert(idx < _track_ids.size());
  return _track_ids[idx];
}

MappointPtr Frame::GetMappoint(size_t idx){
  assert(idx < _mappoints.size());
  return _mappoints[idx];
}

std::vector<MappointPtr>& Frame::GetAllMappoints(){
  return _mappoints;
}

/*
将三角化后的world坐标系下的3D点插入到mappoints中
*/
void Frame::InsertMappoint(size_t idx, MappointPtr mappoint){
  assert(idx < FeatureNum());
  _mappoints[idx] = mappoint;
}

//根据左图特征点xy，恢复深度，算出3D点（相机坐标系下）
bool Frame::BackProjectPoint(size_t idx, Eigen::Vector3d& p3D){
  if(idx >= _depth.size() || _depth[idx] <= 0) return false;//判断是不是和右图匹配上了
  Eigen::Vector3d p2D;
  if(!GetKeypointPosition(idx, p2D)) return false;
  if(!_camera->BackProjectStereo(p2D, p3D)) return false;
  return true;
}

CameraPtr Frame::GetCamera(){
  return _camera;
}

void Frame::FindNeighborKeypoints(Eigen::Vector3d& p2D, std::vector<int>& indices, double r, bool filter) const{
  double x = p2D(0);
  double y = p2D(1);
  double xr = p2D(2);
  const int min_grid_x = std::max(0, (int)std::floor((x-r)*_grid_width_inv));
  const int max_grid_x = std::min((int)(FRAME_GRID_COLS-1), (int)std::ceil((x+r)*_grid_width_inv));
  const int min_grid_y = std::max(0, (int)std::floor((y-r)*_grid_height_inv));
  const int max_grid_y = std::min((int)(FRAME_GRID_ROWS-1), (int)std::ceil((y+r)*_grid_height_inv));
  if(min_grid_x >= FRAME_GRID_COLS || max_grid_x < 0 || min_grid_y >= FRAME_GRID_ROWS || max_grid_y <0) return;

  for(int gx = min_grid_x; gx <= max_grid_x; gx++){
    for(int gy = min_grid_y; gy <= max_grid_y; gy++){
      if(_feature_grid[gx][gy].empty()) continue;
      for(auto& idx : _feature_grid[gx][gy]){
        if(filter && _mappoints[idx] && !_mappoints[idx]->IsBad()) continue;

        const double dx = _keypoints[idx].pt.x - x;
        const double dy = _keypoints[idx].pt.y - y;
        const double dxr = (xr > 0) ? (_u_right[idx] - xr) : 0;
        if(std::abs(dx) < r && std::abs(dy) < r && std::abs(dxr) < r){
          indices.push_back(idx);
        }
      }
    }
  }
}

size_t Frame::LineNum(){
  return _lines.size();
}

/*
给frame的成员变量_line_track_ids设置line_track_id的顺序
*/
void Frame::SetLineTrackId(size_t idx, int line_track_id){
  if(idx < _lines.size()){
    _line_track_ids[idx] = line_track_id;
  }
}

int Frame::GetLineTrackId(size_t idx){
  if(idx < _lines.size()){
    return _line_track_ids[idx];
  }else{
    return -1;
  }
}

bool Frame::GetLine(size_t idx, Eigen::Vector4d& line){
  if(idx >= _lines.size()) return false;

  line = _lines[idx];
  return true;
}

bool Frame::GetLineRight(size_t idx, Eigen::Vector4d& line){
  if(idx >= _lines.size() || !_lines_right_valid[idx]) return false;
  
  line = _lines_right[idx];
  return true;
}

const std::vector<int>& Frame::GetAllLineTrackId(){
  return _line_track_ids;
}

const std::vector<Eigen::Vector4d>& Frame::GatAllLines(){
  return _lines;
}

// const std::vector<Eigen::Vector4d>& Frame::GatAllLinesbeforefilter(){
//   return _lines_before_filter;
// }

const std::vector<Eigen::Vector4d>& Frame::GatAllRightLines(){
  return _lines_right;
}

bool Frame::GetRightLineStatus(size_t idx){
  if(idx >=0 && idx < _lines_right_valid.size()){
    return _lines_right_valid[idx];
  }
  return false;
}

const std::vector<bool>& Frame::GetAllRightLineStatus(){
  return _lines_right_valid;
}

void Frame::InsertMapline(size_t idx, MaplinePtr mapline){
  if(idx < _lines.size()){
    _maplines[idx] = mapline;
  }
}

std::vector<MaplinePtr>& Frame::GetAllMaplines(){
  return _maplines;
}

const std::vector<MaplinePtr>& Frame::GetConstAllMaplines(){
  return _maplines;
}

std::map<int, double> Frame::GetPointsOnLine(size_t idx){
  if(idx >= _points_on_lines.size()){
    std::map<int, double> null_map;
    return null_map;
  }
  return _points_on_lines[idx];
}

const std::vector<std::map<int, double>>& Frame::GetPointsOnLines(){
  return _points_on_lines;
}

/*
使用TriangulateByStereo函数传递左线和对应的右线端点像素，然后三角化端点，再把world坐标系下的端点传递给endpoints
*/
bool Frame::TriangulateStereoLine(size_t idx, Vector6d& endpoints){
  return false;
  if(idx >= _lines.size() || !_lines_right_valid[idx]) return false;
  return TriangulateByStereo(_lines[idx], _lines_right[idx], _pose, _camera, endpoints);
}

void Frame::RemoveMapline(MaplinePtr mapline){
  RemoveMapline(mapline->GetLineIdx(_frame_id));
}

void Frame::RemoveMapline(int idx){
  if(idx < _maplines.size() && idx >=0){
    _maplines[idx] = nullptr;
  }
}

void Frame::AddConnection(std::shared_ptr<Frame> frame, int weight){
  std::map<std::shared_ptr<Frame>, int>::iterator it = _connections.find(frame);
  bool add_connection = (it == _connections.end());
  bool change_connection = (!add_connection && _connections[frame] != weight);
  if(add_connection || change_connection){
    if(change_connection){
      _ordered_connections.erase(std::pair<int, std::shared_ptr<Frame>>(it->second, frame));
    }
    _connections[frame] = weight;
    _ordered_connections.insert(std::pair<int, std::shared_ptr<Frame>>(weight, frame));
  }
}

void Frame::AddConnection(std::set<std::pair<int, std::shared_ptr<Frame>>> connections){
  _ordered_connections = connections;
  _connections.clear();
  for(auto& kv : connections){
    _connections[kv.second] = kv.first;
  }
}

void Frame::SetParent(std::shared_ptr<Frame> parent){
  _parent = parent;
}

std::shared_ptr<Frame> Frame::GetParent(){
  return _parent;
}

void Frame::SetChild(std::shared_ptr<Frame> child){
  _child = child;
}

std::shared_ptr<Frame> Frame::GetChild(){
  return _child;
}

void Frame::RemoveConnection(std::shared_ptr<Frame> frame){
  std::map<std::shared_ptr<Frame>, int>::iterator it = _connections.find(frame);
  if(it != _connections.end()){
    _ordered_connections.erase(std::pair<int, std::shared_ptr<Frame>>(it->second, it->first));
    _connections.erase(it);
  }
}

void Frame::RemoveMappoint(MappointPtr mappoint){
  RemoveMappoint(mappoint->GetKeypointIdx(_frame_id));
}

void Frame::RemoveMappoint(int idx){
  if(idx < _mappoints.size() && idx >=0){
    _mappoints[idx] = nullptr;
  }
}

void Frame::DecreaseWeight(std::shared_ptr<Frame> frame, int weight){
  std::map<std::shared_ptr<Frame>, int>::iterator it = _connections.find(frame);
  if(it != _connections.end()){
    _ordered_connections.erase(std::pair<int, std::shared_ptr<Frame>>(it->second, it->first));
    int original_weight = it->second;
    bool to_remove = (original_weight < (weight+5) && _connections.size() >= 2) || (original_weight <= weight);
    if(to_remove){
      _connections.erase(it);
    }else{
      it->second = original_weight - weight;
      _ordered_connections.insert(std::pair<int, std::shared_ptr<Frame>>(it->second, it->first));
    }
  }
}

std::vector<std::pair<int, std::shared_ptr<Frame>>> Frame::GetOrderedConnections(int number){
  int n = (number > 0 && number < _ordered_connections.size()) ? number : _ordered_connections.size();
  return std::vector<std::pair<int, std::shared_ptr<Frame>>>(_ordered_connections.begin(), std::next(_ordered_connections.begin(), n));
}

void Frame::SetPreviousFrame(std::shared_ptr<Frame> previous_frame){
  _previous_frame = previous_frame;
}

std::shared_ptr<Frame> Frame::PreviousFrame(){
  return _previous_frame;
}