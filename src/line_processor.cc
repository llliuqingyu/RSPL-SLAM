#include "line_processor.h"

#include <math.h>
#include <float.h>
#include <iostream>
#include <numeric>

#include "camera.h"
#include "timer.h"

void FilterShortLines(std::vector<Eigen::Vector4f>& lines, float length_thr){
  Eigen::Array4Xf line_array = Eigen::Map<Eigen::Array4Xf, Eigen::Unaligned>(lines[0].data(), 4, lines.size());
  Eigen::ArrayXf length_square = (line_array.row(2) - line_array.row(0)).square() + (line_array.row(3) - line_array.row(1)).square();
  float thr_square = length_thr * length_thr;

  size_t long_line_num = 0;
  for(size_t i = 0; i < lines.size(); i++){
    if(length_square(i) > thr_square){
      lines[long_line_num] = lines[i];
      long_line_num++;
    }
  }
  lines.resize(long_line_num);
}

void FilterShortLines(std::vector<Eigen::Vector4d>& lines, float length_thr){
  Eigen::Array4Xd line_array = Eigen::Map<Eigen::Array4Xd, Eigen::Unaligned>(lines[0].data(), 4, lines.size());
  Eigen::ArrayXd length_square = (line_array.row(2) - line_array.row(0)).square() + (line_array.row(3) - line_array.row(1)).square();
  float thr_square = length_thr * length_thr;

  size_t long_line_num = 0;
  for(size_t i = 0; i < lines.size(); i++){
    if(length_square(i) > thr_square){
      lines[long_line_num] = lines[i];
      long_line_num++;
    }
  }
  lines.resize(long_line_num);
}

float PointLineDistance(Eigen::Vector4f line, Eigen::Vector2f point){
  float x0 = point(0);
  float y0 = point(1);
  float x1 = line(0);
  float y1 = line(1);
  float x2 = line(2);
  float y2 = line(3);
  float d = (std::fabs((y2 - y1) * x0 +(x1 - x2) * y0 + ((x2 * y1) -(x1 * y2)))) / (std::sqrt(std::pow(y2 - y1, 2) + std::pow(x1 - x2, 2)));
  return d;
}

double CVPointLineDistance3D(const std::vector<cv::Point3f> points, const cv::Vec6f& line, std::vector<float>& dist){
  float px = line[3], py = line[4], pz = line[5];
  float vx = line[0], vy = line[1], vz = line[2];
  dist.resize(points.size());
  double sum_dist = 0.;
  float x, y, z;
  Eigen::Vector3f p;

  for(int j = 0; j < points.size(); j++){
    x = points[j].x - px;
    y = points[j].y - py;
    z = points[j].z - pz;

   // cross 
    p(0) = vy * z - vz * y;
    p(1) = vz * x - vx * z;
    p(2) = vx * y - vy * x;

    dist[j] = p.norm();
    sum_dist += dist[j];
  }

  return sum_dist;
}

void EigenPointLineDistance3D(
    const std::vector<Eigen::Vector3d>& points, const Vector6d& line, std::vector<double>& dist){
  Eigen::Vector3d pl = line.head(3);
  Eigen::Vector3d v = line.tail(3);
  Eigen::Vector3d dp;
  Eigen::Vector3d p;
  size_t point_num = points.size();
  dist.resize(point_num);
  for(size_t i = 0; i < point_num; i++){
    dp = points[i] - pl;
    p = v.cross(dp);
    dist[i] = p.norm();
  }
}

float AngleDiff(float& angle1, float& angle2){
  float d_angle_case1 = std::abs(angle2 - angle1);
  float d_angle_case2 = M_PI + std::min(angle1, angle2) - std::max(angle1, angle2);
  return std::min(d_angle_case1, d_angle_case2);
}

Eigen::Vector4f MergeTwoLines(const Eigen::Vector4f& line1, const Eigen::Vector4f& line2){
  double xg = 0.0, yg = 0.0;
  double delta1x = 0.0, delta1y = 0.0, delta2x = 0.0, delta2y = 0.0;
  float ax = 0, bx = 0, cx = 0, dx = 0;
  float ay = 0, by = 0, cy = 0, dy = 0;
  double li = 0.0, lj = 0.0;
  double thi = 0.0, thj = 0.0, thr = 0.0;
  double axg = 0.0, bxg = 0.0, cxg = 0.0, dxg = 0.0, delta1xg = 0.0, delta2xg = 0.0;

  ax = line1(0);
  ay = line1(1);
  bx = line1(2);
  by = line1(3);

  cx = line2(0);
  cy = line2(1);
  dx = line2(2);
  dy = line2(3);

  float dlix = (bx - ax);
  float dliy = (by - ay);
  float dljx = (dx - cx);
  float dljy = (dy - cy);

  li = sqrt((double) (dlix * dlix) + (double) (dliy * dliy));
  lj = sqrt((double) (dljx * dljx) + (double) (dljy * dljy));

  xg = (li * (double) (ax + bx) + lj * (double) (cx + dx))
      / (double) (2.0 * (li + lj));
  yg = (li * (double) (ay + by) + lj * (double) (cy + dy))
      / (double) (2.0 * (li + lj));

  if(dlix == 0.0f) thi = CV_PI / 2.0;
  else thi = atan(dliy / dlix);

  if(dljx == 0.0f) thj = CV_PI / 2.0;
  else thj = atan(dljy / dljx);

  if (fabs(thi - thj) <= CV_PI / 2.0){
      thr = (li * thi + lj * thj) / (li + lj);
  }
  else{
      double tmp = thj - CV_PI * (thj / fabs(thj));
      thr = li * thi + lj * tmp;
      thr /= (li + lj);
  }

  axg = ((double) ay - yg) * sin(thr) + ((double) ax - xg) * cos(thr);
  bxg = ((double) by - yg) * sin(thr) + ((double) bx - xg) * cos(thr);
  cxg = ((double) cy - yg) * sin(thr) + ((double) cx - xg) * cos(thr);
  dxg = ((double) dy - yg) * sin(thr) + ((double) dx - xg) * cos(thr);

  delta1xg = std::min(axg, std::min(bxg, std::min(cxg,dxg)));
  delta2xg = std::max(axg, std::max(bxg, std::max(cxg,dxg)));

  delta1x = delta1xg * std::cos(thr) + xg;
  delta1y = delta1xg * std::sin(thr) + yg;
  delta2x = delta2xg * std::cos(thr) + xg;
  delta2y = delta2xg * std::sin(thr) + yg;

  Eigen::Vector4f new_line;
  new_line << (float)delta1x, (float)delta1y, (float)delta2x, (float)delta2y;
  return new_line;
}

void AssignPointsToLines(std::vector<Eigen::Vector4d>& lines, Eigen::Matrix<double, 259, Eigen::Dynamic>& points, 
    std::vector<std::map<int, double>>& relation){
  Eigen::Array2Xd point_array = points.middleRows(1, 2).array();//提取x,y坐标
  Eigen::Array4Xd line_array = Eigen::Map<Eigen::Array4Xd, Eigen::Unaligned>(lines[0].data(), 4, lines.size());
  //写的这么花里胡哨，其实就是将lines的4个点提取出来，存到line_array里

  Eigen::ArrayXd x = point_array.row(0);
  Eigen::ArrayXd y = point_array.row(1); 

  Eigen::ArrayXd x1 = line_array.row(0);
  Eigen::ArrayXd y1 = line_array.row(1);
  Eigen::ArrayXd x2 = line_array.row(2);
  Eigen::ArrayXd y2 = line_array.row(3);

  Eigen::ArrayXd A = y2 - y1;
  Eigen::ArrayXd B = x1 - x2;
  Eigen::ArrayXd C = x2 * y1 - x1 * y2;
  Eigen::ArrayXd D = (A.square() + B.square()).sqrt();//直线的一般方程

  relation.clear();
  relation.reserve(lines.size());
  for(int i = 0, line_num = lines.size(); i < line_num; i++){
    std::map<int, double> points_on_line;
    for(int j = 0, point_num = points.cols(); j < point_num; j++){
      // filter by x, y
      double lx1 = x1(i);
      double ly1 = y1(i);
      double lx2 = x2(i);
      double ly2 = y2(i);
      double px = x(j);
      double py = y(j);
      
      double min_lx = lx1;
      double max_lx = lx2;
      double min_ly = ly1;
      double max_ly = ly2;
      if(lx1 > lx2) std::swap(min_lx, max_lx);
      if(ly1 > ly2) std::swap(min_ly, max_ly);
      if(px < min_lx - 3 || px > max_lx + 3 || py < min_ly - 3 || py > max_ly + 3) continue;// 点要落在线段端点投影内

      // check distance
      float pl_distance = std::abs((A(i) * px + B(i) * py + C(i))) / D(i);
      if(pl_distance > 6) continue;// 如果点离线比较远，则不考虑

      double side1 = std::pow((lx1 - px), 2) + std::pow((ly1 - py), 2);
      double side2 = std::pow((lx2 - px), 2) + std::pow((ly2 - py), 2);
      double line_side = std::pow(D(i), 2);
      if(side1 <= 9 || side2 <= 9 || ((side1 < line_side + side2) && (side2 < line_side + side1))){
        points_on_line[j] = pl_distance;
      }
    }
    relation.push_back(points_on_line);//如果一个点都没找到，该线岂不是会被push_back一个空的map？
  }
}

/*
形参：左图点线归属map，右图点线归属map，点匹配关系，左图点数量，右图点数量，输出线匹配关系
*/
void MatchLines(const std::vector<std::map<int, double>>& points_on_line0, 
    const std::vector<std::map<int, double>>& points_on_line1, const std::vector<cv::DMatch>& point_matches, 
    size_t point_num0, size_t point_num1, std::vector<int>& line_matches){
  size_t line_num0 = points_on_line0.size();// 左图线数量
  size_t line_num1 = points_on_line1.size();// 右图线数量
  line_matches.clear();
  line_matches.resize(line_num0);//以左边为主，初始化匹配关系vector的大小
  for(size_t i = 0; i < line_num0; i++){
    line_matches[i] = -1;//初始化匹配关系vector，全部-1
  }
  if(point_num0 == 0 || point_num1 == 0 || line_num0 == 0 || line_num1 == 0) return;

  std::vector<std::vector<int>> assigned_lines0, assigned_lines1;//这个表格外层是所有特征点，内层是根据map得到的点对应的线的索引
  assigned_lines0.resize(point_num0);//设置外层vector大小
  assigned_lines1.resize(point_num1);
  for(size_t i = 0; i < points_on_line0.size(); i++){//遍历每一条线的map
    for(auto& kv : points_on_line0[i]){//points_on_line0[i]是一个map，kv是map的键值对
      assigned_lines0[kv.first].push_back(i);// kv.first是点的索引，i是线的索引
    }
  }
  
  for(size_t i = 0; i < points_on_line1.size(); i++){
    for(auto& kv : points_on_line1[i]){
      assigned_lines1[kv.first].push_back(i);
    }
  }

  // fill in matching matrix
  Eigen::MatrixXi matching_matrix = Eigen::MatrixXi::Zero(line_num0, line_num1);//创建一个line_num0行，line_num1列的矩阵，元素全部为0
  for(auto& point_match : point_matches){
    int idx0 = point_match.queryIdx;
    int idx1 = point_match.trainIdx;

    for(auto& l0 : assigned_lines0[idx0]){
      for(auto& l1 : assigned_lines1[idx1]){
        matching_matrix(l0, l1) += 1;//如果点匹配成功，则对应的线匹配矩阵的元素加1。如果
      }
    }
  }

  // find good matches
  int line_match_num = 0;
  std::vector<int> row_max_value(line_num0), col_max_value(line_num1);
  std::vector<Eigen::VectorXi::Index> row_max_location(line_num0), col_max_location(line_num1);
  for(size_t i = 0; i < line_num0; i++){
    row_max_value[i] = matching_matrix.row(i).maxCoeff(&row_max_location[i]);//将第 i 行的最大值赋值给 row_max_value 向量的第 i 个元素，同时将最大值的索引赋值给 row_max_location 向量的第 i 个元素
  }
  for(size_t j = 0; j < line_num1; j++){
    Eigen::VectorXi::Index col_max_location;
    int col_max_val = matching_matrix.col(j).maxCoeff(&col_max_location);//col_max_val是第 j 列的最大值，col_max_location是最大值处的索引即行索引
    if(col_max_val < 2 || row_max_location[col_max_location] != j) continue;//如果共同匹配点较少，或者行索引与列索引不匹配，则跳过

    // 除以匹配点的个数，这有啥用吗
    float score = (float)(col_max_val * col_max_val) / std::min(points_on_line0[col_max_location].size(), points_on_line1[j].size());
    if(score < 0.8) continue;

    line_matches[col_max_location] = j;//col_max_location是行索引，j是列索引，表示左图的第 col_max_location 条线与右图的第 j 条线匹配
    line_match_num++;
  }
  // std::cout<< "left line num: " << line_num0 << ", right line num: " << line_num1 << ", matched lines: " << line_match_num << std::endl;
  ROS_INFO("\033[32m left line num: %zu, right line num: %zu, matched lines: %d", 
      line_num0, line_num1, line_match_num);
}

void SortPointsOnLine(std::vector<Eigen::Vector2d>& points, std::vector<size_t>& order, bool sort_by_x){
  size_t num_points = points.size();
  if(num_points < 1) return;

  order.clear();
  order.resize(num_points);
  std::iota(order.begin(), order.end(), 0);       
  if(sort_by_x){
    std::sort(order.begin(), order.end(), [&points](size_t i1, size_t i2) { return points[i1](0) < points[i2](0); });
  }else{
    std::sort(order.begin(), order.end(), [&points](size_t i1, size_t i2) { return points[i1](1) < points[i2](1); });
  }                                
}

bool TriangulateByStereo(const Eigen::Vector4d& line_left, const Eigen::Vector4d& line_right, 
    const Eigen::Matrix4d& Twc, const CameraPtr& camera, Vector6d& line_3d){
  double x11 = line_left(0);
  double y11 = line_left(1);
  double x12 = line_left(2);
  double y12 = line_left(3);

  double x21 = line_right(0);
  double y21 = line_right(1);
  double x22 = line_right(2);
  double y22 = line_right(3);

  double dx_left = x12 - x11;
  if(std::abs(dx_left) <= 1e-5) return false;       // parallax is too small
  double dy_left = y12 - y11;
  double angle_left = std::atan(dy_left / dx_left); 
  if(std::abs(angle_left) < 0.087) return false;    // horizontal line，垂直的线不要，为什么？

  double k_inv = dx_left / dy_left;//斜率的倒数
  double x21_left = x11 + k_inv * (y21 - y11);
  double x22_left = x11 + k_inv * (y22 - y11);
  double dx2_left = x22_left - x21_left;
  if(std::abs(dx2_left) <= 1e-5) return false; //还是不要垂直线

  std::vector<Eigen::Vector2d> points;
  points.emplace_back(x11, y11);
  points.emplace_back(x12, y12);
  points.emplace_back(x21_left, y21);//跟求点的时候一样，得到左图的xy坐标，再找到右图中对应的x坐标
  points.emplace_back(x22_left, y22);
  std::vector<size_t> order;
  SortPointsOnLine(points, order);

  size_t i1 = order[0];
  size_t i2 = order[(order.size() - 1)];
  Eigen::Vector3d point_2d1, point_2d2;
  Eigen::Vector3d point_3d1, point_3d2;

  double rate = (x22 - x21) / dx2_left;
  double xr1 = x21 + rate * (points[i1](0) - x21_left);
  double xr2 = x21 + rate * (points[i2](0) - x21_left);

  point_2d1 << points[i1], xr1;
  point_2d2 << points[i2], xr2;

  double dx1 = point_2d1(0) - point_2d1(2);
  double dx2 = point_2d2(0) - point_2d2(2);
  double min_x_diff = camera->MinXDiff();
  double max_x_diff = camera->MaxXDiff();
  if(dx1 < min_x_diff || dx1 > max_x_diff || dx2 < min_x_diff || dx2 > max_x_diff) return false;

  camera->BackProjectStereo(point_2d1, point_3d1);
  camera->BackProjectStereo(point_2d2, point_3d2);

  // form camera to world
  Eigen::Matrix3d Rwc = Twc.block<3, 3>(0, 0);
  Eigen::Vector3d twc = Twc.block<3, 1>(0, 3);
  point_3d1 = Rwc * point_3d1 + twc;
  point_3d2 = Rwc * point_3d2 + twc;
  line_3d << point_3d1, point_3d2;

  return true;
}

bool CompoutePlaneFromPoints(const Eigen::Vector3d& point1, const Eigen::Vector3d& point2, 
    const Eigen::Vector3d& point3, Eigen::Vector4d& plane){
  Eigen::Vector3d line12 = point2 - point1;
  Eigen::Vector3d line13 = point3 - point1;
  Eigen::Vector3d n = line12.cross(line13);
  plane.head(3) = n.normalized();
  plane(3) = - n.transpose() * point1;
  return true;
}

bool ComputeLineFramePlanes(const Eigen::Vector4d& plane1, const Eigen::Vector4d& plane2, Line3DPtr line_3d){
  Eigen::Vector3d n1 = plane1.head(3);
  Eigen::Vector3d n2 = plane2.head(3);

  double cos_theta = n1.transpose() * n2;
  cos_theta /= (n1.norm() * n2.norm());

  // cos10 = cos170 = 0.9848
  // if(std::abs(cos_theta) > 0.9848) return false;

  Eigen::Vector3d d = n1.cross(n2);
  Eigen::Vector3d w = plane2(3) * n1 - plane1(3) * n2;
  line_3d->setD(d);
  line_3d->setW(w);
  line_3d->normalize();
  return true;
}

bool TriangulateByTwoFrames(const Eigen::Vector4d& line_2d1, const Eigen::Matrix4d& pose1, 
    const Eigen::Vector4d& line_2d2, const Eigen::Matrix4d& pose2, const CameraPtr& camera, Line3DPtr line_3d){
  Eigen::Matrix3d Rw1 = pose1.block<3, 3>(0, 0);
  Eigen::Vector3d tw1 = pose1.block<3, 1>(0, 3);
  Eigen::Matrix3d Rw2 = pose2.block<3, 3>(0, 0);
  Eigen::Vector3d tw2 = pose2.block<3, 1>(0, 3);

  Eigen::Matrix3d R12 = Rw1.transpose() * Rw2;
  Eigen::Vector3d t12 = Rw1.transpose() * (tw2 - tw1);

  Eigen::Vector4d plane1, plane2;
  Eigen::Vector3d point11, point12, point13;
  camera->BackProjectMono(line_2d1.head(2), point11);
  camera->BackProjectMono(line_2d1.tail(2), point12);
  point13 << 0.0, 0.0, 0.0;
  if(!CompoutePlaneFromPoints(point11, point12, point13, plane1)) return false;

  Eigen::Vector3d point21, point22;
  camera->BackProjectMono(line_2d2.head(2), point21);
  camera->BackProjectMono(line_2d2.tail(2), point22);

  point21 = R12 * point21 + t12;
  point22 = R12 * point22 + t12;
  if(!CompoutePlaneFromPoints(point21, point22, t12, plane2)) return false;

  bool success = ComputeLineFramePlanes(plane1, plane2, line_3d);

  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  T.rotate(Rw1);
  T.pretranslate(tw1);
  g2o::Line3D line_3d_w = T * (*line_3d);
  line_3d_w.normalize();
  line_3d->setW(line_3d_w.w());
  line_3d->setD(line_3d_w.d());
  return success;
}

bool ComputeLine3DFromEndpoints(const Vector6d& endpoints, Line3DPtr line_3d){
  Eigen::Vector3d point3d1 = endpoints.head(3);//第一个点坐标
  Eigen::Vector3d point3d2 = endpoints.tail(3);//第二个点坐标

  Eigen::Vector3d l = point3d2 - point3d1;//点向量
  if(l.norm() < 0.01) return false;

  Vector6d line_cart;
  line_cart << point3d1, l;
  g2o::Line3D line = g2o::Line3D::fromCartesian(line_cart);//用于从笛卡尔坐标系中的两个点构造一个三维线段。使用 Plücker 坐标表示线段

  line_3d->setW(line.w());// 设置权重矩阵  G2O_TYPES_SLAM3D_ADDONS_API inline Vector3 w() const { return head<3>(); }
  line_3d->setD(line.d());// 设置信息矩阵  G2O_TYPES_SLAM3D_ADDONS_API inline Vector3 d() const { return tail<3>(); }
  return true;
}

bool Point2DTo3D(const Eigen::Vector3d& anchor_point_3d1, const Eigen::Vector3d& anchor_point_3d2, 
  	const Eigen::Vector2d& anchor_point_2d1, const Eigen::Vector2d& anchor_point_2d2, 
    const Eigen::Vector2d& p2D, Eigen::Vector3d& p3D){
  Eigen::Vector2d anchor_line2d = anchor_point_2d2 - anchor_point_2d1;
  anchor_line2d.normalize();
  size_t md = std::abs(anchor_line2d(0)) > std::abs(anchor_line2d(1)) ? 0 : 1;

  double rate = (p2D(md) - anchor_point_2d1(md)) / (anchor_point_2d2(md) - anchor_point_2d1(md));
  p3D = anchor_point_3d1 + rate * (anchor_point_3d2 - anchor_point_3d1);
  return true;
}

LineDetector::LineDetector(const LineDetectorConfig &line_detector_config): _line_detector_config(line_detector_config){
  fld = cv::ximgproc::createFastLineDetector(line_detector_config.length_threshold, line_detector_config.distance_threshold, 
      line_detector_config.canny_th1, line_detector_config.canny_th2, line_detector_config.canny_aperture_size, false);
}

void LineDetector::LineExtractor(const cv::Mat& image, std::vector<Eigen::Vector4d>& lines){
  std::vector<Eigen::Vector4f> source_lines, dst_lines;
  std::vector<cv::Vec4f> cv_lines;
  cv::Mat smaller_image;
  // std::cout << "\033[31m LineExtractor Problem????  \033[0m" << std::endl;
  cv::resize(image, smaller_image, cv::Size(), 0.5, 0.5, cv::INTER_LINEAR);
  fld->detect(smaller_image, cv_lines);

  for(auto& cv_line : cv_lines){
    source_lines.emplace_back(cv_line[0]*2, cv_line[1]*2, cv_line[2]*2, cv_line[3]*2);
    // source_lines.emplace_back(cv_line[0], cv_line[1], cv_line[2], cv_line[3]);
  }

  if(_line_detector_config.do_merge){
    std::vector<Eigen::Vector4f> tmp_lines;
    MergeLines(source_lines, tmp_lines, 0.05, 5, 15);
    FilterShortLines(tmp_lines, 30);
    MergeLines(tmp_lines, dst_lines, 0.03, 3, 50);
    FilterShortLines(dst_lines, 60);

    for(auto& line : dst_lines){
      lines.push_back(line.cast<double>());
    }
  }else{
    for(auto& line : source_lines){
      lines.push_back(line.cast<double>());
    }
  }
  std::cout << "raw LineDetector: " << cv_lines.size() << ", " << "after filtering: " << lines.size() << std::endl;

}

void LineDetector::MergeLines(std::vector<Eigen::Vector4f>& source_lines, std::vector<Eigen::Vector4f>& dst_lines,
    float angle_threshold, float distance_threshold, float endpoint_threshold){

  size_t source_line_num = source_lines.size();
  Eigen::Array4Xf line_array = Eigen::Map<Eigen::Array4Xf, Eigen::Unaligned>(source_lines[0].data(), 4, source_lines.size());
  Eigen::ArrayXf x1 = line_array.row(0);
  Eigen::ArrayXf y1 = line_array.row(1);
  Eigen::ArrayXf x2 = line_array.row(2);
  Eigen::ArrayXf y2 = line_array.row(3);

  Eigen::ArrayXf dx = x2 - x1;
  Eigen::ArrayXf dy = y2 - y1;
  Eigen::ArrayXf eigen_angles = (dy / dx).atan();
  Eigen::ArrayXf length = (dx * dx + dy * dy).sqrt();

  std::vector<float> angles(&eigen_angles[0], eigen_angles.data()+eigen_angles.cols()*eigen_angles.rows());
  std::vector<size_t> indices(angles.size());                                                        
  std::iota(indices.begin(), indices.end(), 0);                                                      
  std::sort(indices.begin(), indices.end(), [&angles](size_t i1, size_t i2) { return angles[i1] < angles[i2]; });

  // search clusters
  float angle_thr = angle_threshold;
  float distance_thr = distance_threshold;
  float ep_thr = endpoint_threshold * endpoint_threshold;
  float quater_PI = M_PI / 4.0;

  std::vector<std::vector<size_t>> neighbors;
  neighbors.resize(source_line_num);
  std::vector<bool> sort_by_x;
  for(size_t i = 0; i < source_line_num; i++){
    size_t idx1 = indices[i];
    float x11 = source_lines[idx1](0);
    float y11 = source_lines[idx1](1);
    float x12 = source_lines[idx1](2);
    float y12 = source_lines[idx1](3);
    float angle1 = angles[idx1];
    bool to_sort_x = (std::abs(angle1) < quater_PI);
    sort_by_x.push_back(to_sort_x);
    if((to_sort_x && (x12 < x11)) || ((!to_sort_x) && y12 < y11)){
      std::swap(x11, x12);
      std::swap(y11, y12);
    }

    for(size_t j = i +1; j < source_line_num; j++){
      size_t idx2 = indices[j];
      float x21 = source_lines[idx2](0);
      float y21 = source_lines[idx2](1);
      float x22 = source_lines[idx2](2);
      float y22 = source_lines[idx2](3);
      if((to_sort_x && (x22 < x21)) || ((!to_sort_x) && y22 < y21)){
        std::swap(x21, x22);
        std::swap(y21, y22);
      }

      // check delta angle
      float angle2 = angles[idx2];
      float d_angle = AngleDiff(angle1, angle2);
      if(d_angle > angle_thr){
        if(std::abs(angle1) < (M_PI_2 - angle_threshold)){
          break;
        }else{
          continue;
        }
      }

      // check distance
      Eigen::Vector2f mid1 = 0.5 * (source_lines[idx1].head(2) + source_lines[idx1].tail(2));
      Eigen::Vector2f mid2 = 0.5 * (source_lines[idx2].head(2) + source_lines[idx2].tail(2));
      float mid1_to_line2 = PointLineDistance(source_lines[idx2], mid1);
      float mid2_to_line1 = PointLineDistance(source_lines[idx1], mid2);
      if(mid1_to_line2 > distance_thr && mid2_to_line1 > distance_thr) continue;

      // check endpoints distance
      float cx12, cy12, cx21, cy21;
      if((to_sort_x && x12 > x22) || (!to_sort_x && y12 > y22)){
        cx12 = x22;
        cy12 = y22;
        cx21 = x11;
        cy21 = y11;
      }else{
        cx12 = x12;
        cy12 = y12;
        cx21 = x21;
        cy21 = y21;
      }
      bool to_merge = ((to_sort_x && cx12 >= cx21) || (!to_sort_x && cy12 >= cy21));
      if(!to_merge){
        float d_ep = (cx21 - cx12) * (cx21 - cx12) + (cy21 - cy12) * (cy21 - cy12);
        to_merge = (d_ep < ep_thr);
      }

      // check cluster code
      if(to_merge){
        neighbors[idx1].push_back(idx2);
        neighbors[idx2].push_back(idx1);
      }
    }
  }

  // clusters
  std::vector<int> cluster_codes(source_line_num, -1);
  std::vector<std::vector<size_t>> cluster_ids;
  for(size_t i = 0; i < source_line_num; i++){
    if(cluster_codes[i] >= 0) continue;

    size_t new_code = cluster_ids.size();
    cluster_codes[i] = new_code;
    std::vector<size_t> to_check_ids = neighbors[i];
    std::vector<size_t> cluster;
    cluster.push_back(i);
    while(to_check_ids.size() > 0){
      std::set<size_t> tmp;
      for(auto& j : to_check_ids){
        if(cluster_codes[j] < 0){
          cluster_codes[j] = new_code;
          cluster.push_back(j);
        }

        std::vector<size_t> j_neighbor = neighbors[j];
        for(auto& k : j_neighbor){
          if(cluster_codes[k] < 0){
            tmp.insert(k);
          } 
        }
      }
      to_check_ids.clear();
      to_check_ids.assign(tmp.begin(), tmp.end());    
    }
    cluster_ids.push_back(cluster);
  }

  // search sub-cluster
  std::vector<std::vector<size_t>> new_cluster_ids;
  for(auto& cluster : cluster_ids){
    size_t cluster_size = cluster.size();
    if(cluster_size <= 2){
      new_cluster_ids.push_back(cluster);
      continue;
    }

    std::sort(cluster.begin(), cluster.end(), [&length](size_t i1, size_t i2) { return length(i1) > length(i2); });
    std::unordered_map<size_t, size_t> line_location;
    for(size_t i = 0; i < cluster_size; i++){
      line_location[cluster[i]] = i;
    }

    std::vector<bool> clustered(cluster_size, false);
    for(size_t j = 0; j < cluster_size; j++){
      if(clustered[j]) continue;

      size_t line_idx = cluster[j];
      std::vector<size_t> sub_cluster;
      sub_cluster.push_back(line_idx);
      std::vector<size_t> line_neighbors = neighbors[line_idx];
      for(size_t k : line_neighbors){
        clustered[line_location[k]] = true;
        sub_cluster.push_back(k);
      }
      new_cluster_ids.push_back(sub_cluster);
    }
  }

  // merge clusters
  dst_lines.clear();
  dst_lines.reserve(new_cluster_ids.size());
  for(auto& cluster : new_cluster_ids){
    size_t idx0 = cluster[0];
    Eigen::Vector4f new_line = source_lines[idx0];
    for(size_t i = 1; i < cluster.size(); i++){
      new_line = MergeTwoLines(new_line, source_lines[cluster[i]]);
    }
    dst_lines.push_back(new_line);
  }
}