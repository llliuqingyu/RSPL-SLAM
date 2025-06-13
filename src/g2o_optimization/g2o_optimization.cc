#include "g2o_optimization/g2o_optimization.h"

#include <fstream>
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/types/sim3/types_seven_dof_expmap.h>

#include "read_configs.h"
#include "g2o_optimization/edge_project_line.h"
#include "g2o_optimization/edge_project_stereo_line.h"

void LocalmapOptimization(MapOfPoses& poses, MapOfPoints3d& points, MapOfLine3d& lines, std::vector<CameraPtr>& camera_list, 
    VectorOfMonoPointConstraints& mono_point_constraints, VectorOfStereoPointConstraints& stereo_point_constraints, 
    VectorOfMonoLineConstraints& mono_line_constraints, VectorOfStereoLineConstraints& stereo_line_constraints, 
    const OptimizationConfig& cfg){
  // Setup optimizer
  typedef g2o::BlockSolver<g2o::BlockSolverTraits<-1, -1> > SlamBlockSolver;
  typedef g2o::LinearSolverEigen<SlamBlockSolver::PoseMatrixType> SlamLinearSolver;

  g2o::SparseOptimizer optimizer;
  optimizer.setVerbose(false);

  auto linear_solver = g2o::make_unique<SlamLinearSolver>();
  linear_solver->setBlockOrdering(false);
  g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(
      g2o::make_unique<SlamBlockSolver>(std::move(linear_solver)));
  optimizer.setAlgorithm(solver);

  // frame vertex
  int max_frame_id = 0;
  for(auto& kv : poses){
    g2o::VertexSE3Expmap* frame_vertex = new g2o::VertexSE3Expmap();
    frame_vertex->setEstimate(g2o::SE3Quat(kv.second.q, kv.second.p).inverse());
    frame_vertex->setId(kv.first);
    frame_vertex->setFixed(kv.second.fixed);
    max_frame_id = std::max(max_frame_id, kv.first);
    optimizer.addVertex(frame_vertex);
  }  
  max_frame_id++;

  // point vertex
  int max_point_id = 0;
  for(auto& kv : points){
    g2o::VertexPointXYZ* point_vertex = new g2o::VertexPointXYZ();
    point_vertex->setEstimate(kv.second.p);
    int point_id = kv.first+max_frame_id;
    point_vertex->setId((point_id));
    max_point_id = std::max(max_point_id, point_id);
    point_vertex->setMarginalized(true);
    optimizer.addVertex(point_vertex);
  }
  max_point_id++;

  // line vertex
  for(auto& kv : lines){
    g2o::VertexLine3D* line_vertex = new g2o::VertexLine3D();
    line_vertex->setEstimateData(kv.second.line_3d);
    line_vertex->setId((max_point_id + kv.first));
    line_vertex->setMarginalized(true);
    optimizer.addVertex(line_vertex);
  }
  
  // point edges
  std::vector<g2o::EdgeSE3ProjectXYZ*> mono_edges; 
  mono_edges.reserve(mono_point_constraints.size());
  std::vector<g2o::EdgeStereoSE3ProjectXYZ*> stereo_edges;
  stereo_edges.reserve(stereo_point_constraints.size());
  const float thHuberMonoPoint = sqrt(cfg.mono_point);
  const float thHuberStereoPoint = sqrt(cfg.stereo_point);

  // mono point edges
  for(MonoPointConstraintPtr& mpc : mono_point_constraints){
    g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();
    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex((mpc->id_point+max_frame_id))));
    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(mpc->id_pose)));
    e->setMeasurement(mpc->keypoint);
    e->setInformation(Eigen::Matrix2d::Identity());
    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
    e->setRobustKernel(rk);
    rk->setDelta(thHuberMonoPoint);
    e->fx = camera_list[mpc->id_camera]->Fx();
    e->fy = camera_list[mpc->id_camera]->Fy();
    e->cx = camera_list[mpc->id_camera]->Cx();
    e->cy = camera_list[mpc->id_camera]->Cy();

    optimizer.addEdge(e);
    mono_edges.push_back(e);
  }

  // stereo point edges
  for(StereoPointConstraintPtr& spc : stereo_point_constraints){
    g2o::EdgeStereoSE3ProjectXYZ* e = new g2o::EdgeStereoSE3ProjectXYZ();

    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex((spc->id_point+max_frame_id))));
    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(spc->id_pose)));
    e->setMeasurement(spc->keypoint);
    e->setInformation(Eigen::Matrix3d::Identity());
    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
    e->setRobustKernel(rk);
    rk->setDelta(thHuberStereoPoint);
    e->fx = camera_list[spc->id_camera]->Fx();
    e->fy = camera_list[spc->id_camera]->Fy();
    e->cx = camera_list[spc->id_camera]->Cx();
    e->cy = camera_list[spc->id_camera]->Cy();
    e->bf = camera_list[spc->id_camera]->BF();

    optimizer.addEdge(e);
    stereo_edges.push_back(e);
  }

  // line edges
  std::vector<EdgeSE3ProjectLine*> mono_line_edges; 
  mono_line_edges.reserve(mono_line_constraints.size());
  std::vector<EdgeStereoSE3ProjectLine*> stereo_line_edges;
  stereo_line_edges.reserve(stereo_line_constraints.size());
  const float thHuberMonoLine = sqrt(cfg.mono_line);
  const float thHuberStereoLine = sqrt(cfg.stereo_line);
  // mono line edges
  for(MonoLineConstraintPtr& mlc : mono_line_constraints){
    EdgeSE3ProjectLine* e = new EdgeSE3ProjectLine();
    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex((mlc->id_line+max_point_id))));
    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(mlc->id_pose)));
    e->setMeasurement(mlc->line_2d);
    e->setInformation(Eigen::Matrix2d::Identity() * 0.1);
    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
    e->setRobustKernel(rk);
    rk->setDelta(thHuberMonoLine);
    double fx = camera_list[mlc->id_camera]->Fx();
    double fy = camera_list[mlc->id_camera]->Fy();
    double cx = camera_list[mlc->id_camera]->Cx();
    double cy = camera_list[mlc->id_camera]->Cy();
    e->fx = fx;
    e->fy = fy;
    e->Kv << -fy * cx, -fx * cy, fx * fy;
    optimizer.addEdge(e);
    mono_line_edges.push_back(e);
  }

  // stereo line edges
  for(StereoLineConstraintPtr& slc : stereo_line_constraints){
    EdgeStereoSE3ProjectLine* e = new EdgeStereoSE3ProjectLine();
    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex((slc->id_line+max_point_id))));
    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(slc->id_pose)));
    e->setMeasurement(slc->line_2d);
    e->setInformation(Eigen::Matrix4d::Identity() * 0.1);
    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
    e->setRobustKernel(rk);
    rk->setDelta(thHuberStereoLine);
    double fx = camera_list[slc->id_camera]->Fx();
    double fy = camera_list[slc->id_camera]->Fy();
    double cx = camera_list[slc->id_camera]->Cx();
    double cy = camera_list[slc->id_camera]->Cy();
    double bf = camera_list[slc->id_camera]->BF();
    e->fx = fx;
    e->fy = fy;
    e->b = bf / fx;
    e->Kv << -fy * cx, -fx * cy, fx * fy;
    optimizer.addEdge(e);
    stereo_line_edges.push_back(e);
  }

  // solve 
  optimizer.initializeOptimization();
  optimizer.optimize(10);

  // check inlier observations
  for(size_t i=0; i < mono_edges.size(); i++){
    g2o::EdgeSE3ProjectXYZ* e = mono_edges[i];
    if(e->chi2() > cfg.mono_point || !e->isDepthPositive()){
      e->setLevel(1);
    }
    e->setRobustKernel(0);
  }

  for(size_t i=0; i < stereo_edges.size(); i++){    
    g2o::EdgeStereoSE3ProjectXYZ* e = stereo_edges[i];
    if(e->chi2() > cfg.stereo_point || !e->isDepthPositive()){
        e->setLevel(1);
    }
    e->setRobustKernel(0);
  }

  for(size_t i=0; i < mono_line_edges.size(); i++){
    EdgeSE3ProjectLine* e = mono_line_edges[i];
    if(e->chi2() > cfg.mono_line){
      e->setLevel(1);
    }
    e->setRobustKernel(0);
  }

  for(size_t i=0; i < stereo_line_edges.size(); i++){    
    EdgeStereoSE3ProjectLine* e = stereo_line_edges[i];
    if(e->chi2() > cfg.stereo_line){
        e->setLevel(1);
    }
    e->setRobustKernel(0);
  }

  // optimize again without the outliers
  optimizer.initializeOptimization(0);
  optimizer.optimize(5);

  // check inlier observations     
  for(size_t i = 0; i < mono_edges.size(); i++){
    g2o::EdgeSE3ProjectXYZ* e = mono_edges[i];
    mono_point_constraints[i]->inlier = (e->chi2() <= cfg.mono_point && e->isDepthPositive());
  }

  for(size_t i = 0; i < stereo_edges.size(); i++){    
    g2o::EdgeStereoSE3ProjectXYZ* e = stereo_edges[i];
    stereo_point_constraints[i]->inlier = (e->chi2() <= cfg.stereo_point && e->isDepthPositive());
  }

  for(size_t i = 0; i < mono_line_edges.size(); i++){
    EdgeSE3ProjectLine* e = mono_line_edges[i];
    mono_line_constraints[i]->inlier = (e->chi2() <= cfg.mono_line);
  }

  for(size_t i = 0; i < stereo_line_edges.size(); i++){    
    EdgeStereoSE3ProjectLine* e = stereo_line_edges[i];
    stereo_line_constraints[i]->inlier = (e->chi2() <= cfg.stereo_line);
  }

  // Recover optimized data
  // Keyframes
  for(MapOfPoses::iterator it = poses.begin(); it!=poses.end(); ++it){
    g2o::VertexSE3Expmap* frame_vertex = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(it->first));
    g2o::SE3Quat SE3quat = frame_vertex->estimate().inverse();
    it->second.p = SE3quat.translation();
    it->second.q = SE3quat.rotation();
  }
  // Points
  for(MapOfPoints3d::iterator it = points.begin(); it!=points.end(); ++it){
    g2o::VertexPointXYZ* point_vertex = static_cast<g2o::VertexPointXYZ*>(optimizer.vertex(it->first+max_frame_id));
    it->second.p = point_vertex->estimate();
  }

  // Lines
  for(MapOfLine3d::iterator it = lines.begin(); it!=lines.end(); ++it){
    g2o::VertexLine3D* line_vertex = static_cast<g2o::VertexLine3D*>(optimizer.vertex(it->first+max_point_id));
    it->second.line_3d = line_vertex->estimate();
  } 
}


// 形参：当前帧poses, 3D点points, 相机列表camera_list, 单目点约束mono_point_constraints, 双目点约束stereo_point_constraints（点约束都是跟上一帧匹配上的）
int FrameOptimization(MapOfPoses& poses, MapOfPoints3d& points, std::vector<CameraPtr>& camera_list, 
    VectorOfMonoPointConstraints& mono_point_constraints, VectorOfStereoPointConstraints& stereo_point_constraints,
    const OptimizationConfig& cfg){
  assert(poses.size() == 1);//若不是1，说明不是单帧优化，即已经是局部地图优化了，直接中断
  g2o::SparseOptimizer optimizer;
  optimizer.setVerbose(false);//设置为false，不输出优化信息
  std::unique_ptr<g2o::BlockSolver_6_3::LinearSolverType> linear_solver;
  linear_solver = g2o::make_unique<g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>>();
  g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(
      g2o::make_unique<g2o::BlockSolver_6_3>(std::move(linear_solver)));
  optimizer.setAlgorithm(solver);

  // frame vertex
  MapOfPoses::iterator pose_it = poses.begin();//这里poses.size() == 1，所以直接取第一个元素，且是指针
  g2o::VertexSE3Expmap* frame_vertex = new g2o::VertexSE3Expmap();//表示三维空间中的位姿顶点，在图优化里代表一个节点，该节点存储的是相机的位姿信息
  frame_vertex->setEstimate(g2o::SE3Quat(pose_it->second.q, pose_it->second.p).inverse());//second表示map成员的值，inverse()表示将位姿转换为逆变换
  // 逆变换是因为在g2o中，顶点的位姿是相对于世界坐标系的，而在这里我们需要将其转换为相对于当前帧的坐标系
  frame_vertex->setId(0);
  frame_vertex->setFixed(false);
  optimizer.addVertex(frame_vertex);

  // point edges
  std::vector<g2o::EdgeSE3ProjectXYZOnlyPose*> mono_edges; 
  mono_edges.reserve(mono_point_constraints.size());
  std::vector<g2o::EdgeStereoSE3ProjectXYZOnlyPose*> stereo_edges;
  stereo_edges.reserve(stereo_point_constraints.size());
  const float deltaMonoPoint = sqrt(cfg.mono_point);
  const float deltaStereoPoint = sqrt(cfg.stereo_point);
  const float deltaMonoLine = sqrt(cfg.mono_line);
  const float deltaStereoLine = sqrt(cfg.stereo_line);

  // mono edges
  for(MonoPointConstraintPtr& mpc : mono_point_constraints){
    Position3d point = points[mpc->id_point];//这里id_point是键值对的key，表示当前帧与上一帧匹配的点的id

    g2o::EdgeSE3ProjectXYZOnlyPose* e = new g2o::EdgeSE3ProjectXYZOnlyPose();//一元边，表示世界点的重投影误差，仅优化相机的位姿
    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
    //0表示连接第0个顶点，因为是一元边，所以是0
    //用于在继承体系中进行安全的向下或向上类型转换。
    //这里把 optimizer.vertex(0) 返回的指针转换为 g2o::OptimizableGraph::Vertex* 类型，以匹配 setVertex 方法的参数要求
    e->setMeasurement(mpc->keypoint);//设置测量值，即当前帧中点的像素坐标（通过superglue匹配以及ransac后的所谓的匹配值）
    e->setInformation(Eigen::Matrix2d::Identity());//设置信息矩阵为单位矩阵，表示对测量值的置信度为1，即对e中的测量值的估计值与真实值之间的误差的方差为1
    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;//鲁棒核函数，用于减少异常值对优化结果的影响
    e->setRobustKernel(rk);
    rk->setDelta(deltaMonoPoint);
    e->fx = camera_list[mpc->id_camera]->Fx();
    e->fy = camera_list[mpc->id_camera]->Fy();
    e->cx = camera_list[mpc->id_camera]->Cx();
    e->cy = camera_list[mpc->id_camera]->Cy();
    e->Xw = point.p;//拿到当前帧与上一帧匹配的点的三维坐标（世界坐标系）

    optimizer.addEdge(e);
    mono_edges.push_back(e);
  }

  // stereo edges 同上
  // 但，
  for(StereoPointConstraintPtr& spc : stereo_point_constraints){
    Position3d point = points[spc->id_point];

    g2o::EdgeStereoSE3ProjectXYZOnlyPose* e = new g2o::EdgeStereoSE3ProjectXYZOnlyPose();
    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
    e->setMeasurement(spc->keypoint);
    e->setInformation(Eigen::Matrix3d::Identity());
    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
    e->setRobustKernel(rk);
    rk->setDelta(deltaStereoPoint);
    e->fx = camera_list[spc->id_camera]->Fx();
    e->fy = camera_list[spc->id_camera]->Fy();
    e->cx = camera_list[spc->id_camera]->Cx();
    e->cy = camera_list[spc->id_camera]->Cy();
    e->bf = camera_list[spc->id_camera]->BF();
    e->Xw = point.p;

    optimizer.addEdge(e);
    stereo_edges.push_back(e);
    // 所以在这批约束点集里，有单目点约束和双目点约束，双目即在当前帧中，匹配到的点有左目和右目两个像素坐标
  }

  // solve
  const int its[4]={10, 10, 10, 10};  

  int num_outlier = 0;
  for(size_t iter = 0; iter < 4; iter++){
    frame_vertex->setEstimate(g2o::SE3Quat(pose_it->second.q, pose_it->second.p).inverse());
    optimizer.initializeOptimization(0);
    optimizer.optimize(its[iter]);//这有意义吗，不都是迭代10次吗？
  
    num_outlier=0;
    for(size_t i = 0; i < mono_edges.size(); i++){
      g2o::EdgeSE3ProjectXYZOnlyPose* e = mono_edges[i];
      if(!mono_point_constraints[i]->inlier){
        e->computeError();
      }

      const float chi2 = e->chi2();//获取边的卡方值，表示重投影误差
      if(chi2 > cfg.mono_point){                
        mono_point_constraints[i]->inlier = false;
        e->setLevel(1);//当边的层级被设置为 1 时，这条边在优化过程中会被暂时忽略，不参与优化计算
        num_outlier++;
      }
      else{
        mono_point_constraints[i]->inlier = true;
        e->setLevel(0);
      }
      // 随着跌带增加，大部分异常值已被识别并通过 e->setLevel(1) 排除在优化过程之外。
      //在第 3 次迭代时，剩余的数据点大多为有效数据，此时移除鲁棒核函数，让优化器基于
      //全部有效数据进行更精确的优化，以提高相机位姿估计的准确性
      if(iter == 2) e->setRobustKernel(0);
    }

    
    for(size_t i = 0; i < stereo_edges.size(); i++){
      g2o::EdgeStereoSE3ProjectXYZOnlyPose* e = stereo_edges[i];
      if(!stereo_point_constraints[i]->inlier){
         e->computeError();
      }

      const float chi2 = e->chi2();
      if(chi2 > cfg.stereo_point){                
        stereo_point_constraints[i]->inlier = false;
        e->setLevel(1);
        num_outlier++;
      }
      else{
        stereo_point_constraints[i]->inlier = true;
        e->setLevel(0);
      }
      if(iter == 2) e->setRobustKernel(0);
    }

    if(optimizer.edges().size()<10) break;//如果优化器中的边数小于10，说明没有足够的约束进行优化，直接跳出循环
  }

  // recover optimized data
  g2o::SE3Quat SE3quat = frame_vertex->estimate().inverse();
  pose_it->second.p = SE3quat.translation();
  pose_it->second.q = SE3quat.rotation();// 然后呢？pose_it也没返回啊（答：pose_it是pose的迭代器，即指针）

  // 也没有过滤边的信息啊，这么奇怪
  return (mono_point_constraints.size() + stereo_point_constraints.size() - num_outlier);
}


// 形参：当前帧，地图点集合（当前帧与上一帧匹配的，并且被初始化了上一帧的mappoint，无效点为nullptr），输出的位姿矩阵，输出的内点索引
// 返回值：内点数量
int SolvePnPWithCV(FramePtr frame, std::vector<MappointPtr>& mappoints, 
    Eigen::Matrix4d& pose, std::vector<int>& inliers){
  std::vector<cv::Point3f> object_points;
  std::vector<cv::Point2f> image_points;
  std::vector<int> point_indexes;
  cv::Mat camera_matrix, dist_coeffs;
  CameraPtr camera = frame->GetCamera();
  camera->GetCamerMatrix(camera_matrix);
  camera->GetDistCoeffs(dist_coeffs);
  cv::Mat rotation_vector;
  cv::Mat translation_vector;
  cv::Mat cv_inliers;

  //总的mappoint集合是：当前帧与上一帧匹配的点集合
  //但是这个点集在上一帧并不一定都有初始化，所以要剔除掉没有初始化的点，也就是mappoint为nullptr的点
  //同时，要剔除掉无效的点，也就是mappoint->IsValid()为false的点
  //同时，要剔除掉当前帧没有对应上的点，也就是frame->GetKeypointPosition(i, keypoint)为false的点
  for(size_t i = 0; i < mappoints.size(); i++){
    MappointPtr mpt = mappoints[i];
    if(mpt == nullptr || !mpt->IsValid()) continue;
    Eigen::Vector3d keypoint; 
    if(!frame->GetKeypointPosition(i, keypoint)) continue;
    const Eigen::Vector3d& point_position = mpt->GetPosition();
    object_points.emplace_back(point_position(0), point_position(1), point_position(2));//这里其实是上一帧的世界坐标系下3d点
    image_points.emplace_back(keypoint(0), keypoint(1));//这里是当前帧的图像平面上的2d点
    point_indexes.emplace_back(i);
  }
  if(object_points.size() < 8) return 0;//至少需要8个点才能进行PnP求解

  try{
    // 特征点在世界坐标系下的 3D 坐标、表示 3D 点在图像平面上对应的 2D 投影点的坐标、相机的内参矩阵、相机的畸变系数（没有畸变，可以传入一个空矩阵 cv::Mat()）
    // 输出的旋转向量、输出的平移向量、是否使用 RANSAC 方法来估计外参、RANSAC 的迭代次数、RANSAC 的阈值、RANSAC 的置信度
    cv::solvePnPRansac(object_points, image_points, camera_matrix, dist_coeffs, 
        rotation_vector, translation_vector, false, 100, 20.0, 0.99, cv_inliers);
  }catch(...){
    return 0;
  }

  cv::Mat cv_Rcw;
  cv::Rodrigues(rotation_vector, cv_Rcw);//旋转向量转换为旋转矩阵
  Eigen::Matrix3d eigen_Rcw;
  Eigen::Vector3d eigen_tcw;
  cv::cv2eigen(cv_Rcw, eigen_Rcw);//cv::Mat转换为Eigen::Matrix3d
  cv::cv2eigen(translation_vector, eigen_tcw);//cv::Mat转换为Eigen::Vector3d
  Eigen::Matrix3d eigen_Rwc = eigen_Rcw.transpose();
  pose.block<3, 3>(0, 0) = eigen_Rwc;
  pose.block<3, 1>(0, 3) = eigen_Rwc * (-eigen_tcw);

  inliers = std::vector<int>(mappoints.size(), -1);//初始化inliers为-1，表示没有内点。这个inliers是所有当前帧与上一帧匹配上的
  for(int i = 0; i < cv_inliers.rows; i++){
    int inlier_idx = cv_inliers.at<int>(i, 0);//获取 cv_inliers 矩阵中第 i 行第 0 列的元素，该元素是 object_points 和 image_points 向量中的内点索引
    int point_idx = point_indexes[inlier_idx];//point_indexes保存的是mappoints中用到的点索引。所以point_indexes[inlier_idx]指的是mappoints中的有效内点的索引
    inliers[point_idx] = mappoints[point_idx]->GetId();//inliers是当前mappoints中有效内点索引，对应的mappoints中的点的id（但点id是不连续的啊）
    //换句话说，这里的inliers是当前帧与上一帧匹配上的点的索引，但对应的是地图中有效的mappoints的id，是为了方便查找？
  }

  int outliers = inliers.size() - cv_inliers.rows;// for debug
  std::cout << "outliers size = " << outliers << std::endl;
  return cv_inliers.rows;
}
