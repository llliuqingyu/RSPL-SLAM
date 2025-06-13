#ifndef MAPLINE_H_
#define MAPLINE_H_

#include <limits>
#include <string>
#include <memory>
#include <unordered_map>
#include <vector>
#include <map>
#include <Eigen/Dense>
#include <Eigen/SparseCore>

#include <g2o/types/slam3d/types_slam3d.h>
#include <g2o/types/slam3d_addons/types_slam3d_addons.h>

#include "utils.h"


class Mapline{
public:
  enum Type {
    UnTriangulated = 0,
    Good = 1,
    Bad = 2,
  };

  Mapline();
  Mapline(int mappoint_id);
  void SetId(int id);
  int GetId();
  void SetType(Type& type);
  Type GetType();
  void SetBad();
  bool IsBad();
  void SetGood();
  bool IsValid();

  void SetEndpoints(Vector6d& p, bool compute_line3d=true);
  Vector6d& GetEndpoints();
  void SetEndpointsValidStatus(bool status);
  bool EndpointsValid();
  void SetEndpointsUpdateStatus(bool status);
  bool ToUpdateEndpoints();
  void SetLine3D(g2o::Line3D& line_3d);
  void SetLine3DPtr(Line3DPtr& line_3d);
  ConstLine3DPtr GetLine3DPtr();
  g2o::Line3D GetLine3D();

  void AddObverser(const int& frame_id, const int& line_index);
  void RemoveObverser(const int& frame_id);
  int ObverserNum();
  const std::map<int, int>& GetAllObversers();
  int GetLineIdx(int frame_id);

  void SetObverserEndpointStatus(int frame_id, int status = 1);
  int GetObverserEndpointStatus(int frame_id);
  const std::map<int, int>& GetAllObverserEndpointStatus();

public:
  int local_map_optimization_frame_id;

private:
  int _id;
  Type _type;
  bool _to_update_endpoints;
  bool _endpoints_valid;
  Vector6d _endpoints;
  Line3DPtr _line_3d;//方向向量和矩，就是plucker坐标      信息矩阵line_processor.cc中的430行
  std::map<int, int> _obversers;  // frame_id - line_index 
  std::map<int, int> _included_endpoints;//存储每个线的状态？源程序这里初始都是0
};

typedef std::shared_ptr<Mapline> MaplinePtr; // 这里只是一个线的信息，可不是一帧上所有的线

#endif  // MAPLINE_H_