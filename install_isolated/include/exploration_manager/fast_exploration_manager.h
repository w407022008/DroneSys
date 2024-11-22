#ifndef _EXPLORATION_MANAGER_H_
#define _EXPLORATION_MANAGER_H_

#include <ros/ros.h>
#include <Eigen/Eigen>
#include <memory>
#include <vector>

using Eigen::Vector3d;
using std::shared_ptr;
using std::unique_ptr;
using std::vector;

namespace fast_planner {
class EDTEnvironment;
class SDFMap;
class FastPlannerManager;
class FrontierFinder;
struct ExplorationParam;
struct ExplorationData;

enum EXPL_RESULT { NO_FRONTIER, FAIL, SUCCEED };

class FastExplorationManager {
public:
  FastExplorationManager();
  ~FastExplorationManager();

  void initialize(ros::NodeHandle& nh);

  int planExploreMotion(const Vector3d& pos, const Vector3d& vel, const Vector3d& acc,
                        const Vector3d& yaw);
  void getPath(vector<Vector3d>& path);
  void getGoal(Vector3d& goal);
  void getViewpoint(Eigen::Vector4d& next_viewpoint);
  void getFrontierInfo(Eigen::Vector4d& next_frontier_info);
  void getFrontier(vector<vector<Vector3d>>& frontier_, vector<vector<Vector3d>>& dead_frontiers_);

  shared_ptr<ExplorationData> ed_;
  shared_ptr<ExplorationParam> ep_;
  shared_ptr<FastPlannerManager> planner_manager_;
  shared_ptr<FrontierFinder> frontier_finder_;
  // unique_ptr<ViewFinder> view_finder_;

private:
  shared_ptr<EDTEnvironment> edt_environment_;
  shared_ptr<SDFMap> sdf_map_;

  Vector3d next_viewpoint_to_visit, next_frontier_average;
  double next_yaw;

  int updateFrontier(const Vector3d& pos);
  void exploreTour(const Vector3d& pos, const Vector3d& vel, const Vector3d& acc, const Vector3d& yaw);
  int planTrajectory(const Vector3d& pos, const Vector3d& vel, const Vector3d& acc, const Vector3d& yaw);
  // Find optimal tour for coarse viewpoints of all frontiers
  void findGlobalTour(const Vector3d& cur_pos, const Vector3d& cur_vel, const Vector3d cur_yaw,
                      vector<int>& indices);

  // Refine local tour for next few frontiers, using more diverse viewpoints
  void refineLocalTour(const Vector3d& cur_pos, const Vector3d& cur_vel, const Vector3d& cur_yaw,
                       const vector<vector<Vector3d>>& n_points, const vector<vector<double>>& n_yaws, const vector<Eigen::Vector3d> n_averages,
                       vector<Vector3d>& refined_pts, vector<double>& refined_yaws, vector<Eigen::Vector3d>& refined_averages);

  void shortenPath(vector<Vector3d>& path);

public:
  typedef shared_ptr<FastExplorationManager> Ptr;
};

}  // namespace fast_planner

#endif
