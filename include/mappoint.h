#ifndef MAPPOINT_H_
#define MAPPOINT_H_

#include <limits>
#include <string>
#include <memory>
#include <unordered_map>
#include <vector>
#include <map>
#include <Eigen/Dense>
#include <Eigen/SparseCore>


class Mappoint{
public:
  enum Type {
    UnTriangulated = 0,
    Good = 1,
    Bad = 2,
  };

  Mappoint();
  Mappoint(int& mappoint_id);
  Mappoint(int& mappoint_id, Eigen::Vector3d& p);
  Mappoint(int& mappoint_id, Eigen::Vector3d& p, Eigen::Matrix<double, 256, 1>& d);
  void SetId(int id);
  int GetId();
  void SetType(Type& type);
  Type GetType();
  void SetBad();
  bool IsBad();
  void SetGood();
  bool IsValid();

  void SetPosition(Eigen::Vector3d& p);
  Eigen::Vector3d& GetPosition();
  void SetDescriptor(const Eigen::Matrix<double, 256, 1>& descriptor);
  Eigen::Matrix<double, 256, 1>& GetDescriptor(); 

  void AddObverser(const int& frame_id, const int& keypoint_index);
  void RemoveObverser(const int& frame_id);
  int ObverserNum();
  std::map<int, int>& GetAllObversers();
  int GetKeypointIdx(int frame_id);

public:
  int tracking_frame_id;
  int last_frame_seen;
  int local_map_optimization_frame_id;

private:
  int _id;
  Type _type;
  Eigen::Vector3d _position;
  Eigen::Matrix<double, 256, 1> _descriptor;
  std::map<int, int> _obversers;  // frame_id - keypoint_index 描述该点被哪个帧看到了，用必要整个容器来处理吗
};

typedef std::shared_ptr<Mappoint> MappointPtr; //这里只是一个点的信息，可不是一帧上所有的点

#endif  // MAPPOINT_H