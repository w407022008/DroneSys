#ifndef _FRONTIER_FINDER_H_
#define _FRONTIER_FINDER_H_

#include <ros/ros.h>
#include <Eigen/Eigen>
#include <memory>
#include <vector>
#include <list>
#include <utility>

using Eigen::Vector3d;
using std::shared_ptr;
using std::unique_ptr;
using std::vector;
using std::list;
using std::pair;

class RayCaster;

namespace fast_planner {
class EDTEnvironment;
class PerceptionUtils;

// Viewpoint to cover a frontier cluster
struct Viewpoint {
  // Position and heading
  Vector3d pos_;
  double yaw_;
  // Fraction of the cluster that can be covered
  // double fraction_;
  int visib_num_;
};

// A frontier cluster, the viewpoints to cover it
struct Frontier {
  // Complete voxels belonging to the cluster
  vector<Vector3d> cells_;
  // down-sampled voxels filtered by voxel grid filter
  vector<Vector3d> filtered_cells_;
  // Average position of all voxels
  Vector3d average_;
  // Idx of cluster
  int id_;
  // Viewpoints that can cover the cluster
  vector<Viewpoint> viewpoints_;
  // Bounding box of cluster, center & 1/2 side length
  Vector3d box_min_, box_max_;
  // Path and cost from this cluster to other clusters
  list<vector<Vector3d>> paths_;
  list<double> costs_;
};

class FrontierFinder {
public:
  FrontierFinder(const shared_ptr<EDTEnvironment>& edt, ros::NodeHandle& nh);
  ~FrontierFinder();

  void updateFrontiers(bool inLocalSpace = true);
  void searchViewpointsOfNewFrontiers();

  void getCellsOfEachFrontier(vector<vector<Vector3d>>& clusters_frontier, vector<vector<Vector3d>>& clusters_dormant_frontier);
  void getBoundingBoxOfEachFrontier(vector<pair<Vector3d, Vector3d>>& boxes);
  // Get viewpoint with highest coverage for each frontier
  void getFarEnoughTopViewpointOfEachFrontierFrom(const Vector3d& cur_pos, vector<Vector3d>& points, vector<double>& yaws,
                            vector<Vector3d>& averages, double min_candidate_dist_);
  void getTopViewpointOfEachFrontier(vector<Vector3d>& points, vector<double>& yaws, vector<Vector3d>& averages);
  // Get several viewpoints for a subset of frontiers
  void getConditionalTopViewpointsFrom_OF(const Vector3d& cur_pos, const vector<int>& ids, const int& view_num,
                         const double& max_decay, vector<vector<Vector3d>>& points,
                         vector<vector<double>>& yaws, double min_candidate_dist_);
  void getTopViewpointsOf_From(const Vector3d& cur_pos, const vector<int>& ids, const int& view_num,
                         const double& max_decay, vector<vector<Vector3d>>& points,
                         vector<vector<double>>& yaws, vector<Eigen::Vector3d>& averages, double min_candidate_dist_);
  void updatePathAndCostAmongTopFrontierViewpoints();
  void computeFullCostMatrixAmongTopFrontierViewpointsFrom(const Vector3d& cur_pos, const Vector3d& cur_vel, const Vector3d cur_yaw,
                         Eigen::MatrixXd& mat);
  void getPathAllFrontiersFrom_Along(const Vector3d& pos, const vector<int>& frontier_ids, vector<Vector3d>& path);
  int countFrontierCells();
  void setNextFrontier(const int& id);
  bool isThereAFrontierCovered();
  void wrapYaw(double& yaw);

  shared_ptr<PerceptionUtils> percep_utils_;

private:
  void splitLargeFrontiers(list<Frontier>& frontiers);
  bool splitHorizontally(const Frontier& frontier, list<Frontier>& splits);
  void mergeFrontiers(Frontier& ftr1, const Frontier& ftr2);
  bool isFrontierChanged(const Frontier& ft);
  bool isFrontierAlmostFullyCovered(const Frontier& ft);
  bool haveOverlap(const Vector3d& min1, const Vector3d& max1, const Vector3d& min2,
                   const Vector3d& max2);
  void computeCellsAverage_CellsBox_FilteredCells_Of(Frontier& frontier);
  void downsample(const vector<Vector3d>& cluster_in, vector<Vector3d>& cluster_out);
  void sampleViewpoints(Frontier& frontier);

  int countVisibleCells(const Vector3d& pos, const double& yaw, const vector<Vector3d>& cluster);
  bool hasNearUnknown(const Vector3d& pos);
  vector<Eigen::Vector3i> sixNeighbors(const Eigen::Vector3i& voxel);
  vector<Eigen::Vector3i> tenNeighbors(const Eigen::Vector3i& voxel);
  vector<Eigen::Vector3i> allNeighbors(const Eigen::Vector3i& voxel);
  bool hasUnknownNeighbor(const Eigen::Vector3i& voxel);
  void expandFrontier(const Eigen::Vector3i& first /* , const int& depth, const int& parent_id */);

  // Wrapper of sdf map
  int Index3DToAddress1D(const Eigen::Vector3i& idx);
  bool knownfree(const Eigen::Vector3i& idx);
  bool inmap(const Eigen::Vector3i& idx);

  // Deprecated
  Eigen::Vector3i searchClearVoxel(const Eigen::Vector3i& pt);
  bool isInBoxes(const vector<pair<Vector3d, Vector3d>>& boxes, const Eigen::Vector3i& idx);
  bool canBeMerged(const Frontier& ftr1, const Frontier& ftr2);
  void findViewpoints(const Vector3d& sample, const Vector3d& ftr_avg, vector<Viewpoint>& vps);

  // Data
  list<Frontier> frontiers_, dormant_frontiers_, tmp_frontiers_;
  vector<char> cell_is_on_frontier_;
  vector<int> removed_ids_;
  list<Frontier>::iterator first_new_ftr_;
  int last_frontier_size, cur_frontier_size;

  // Params
  bool covered_checking_forced_;
  int cluster_min_;
  double cluster_size_xy_, cluster_size_z_;
  double candidate_rmax_, candidate_rmin_, candidate_dphi_, near_unknow_clearance_;
  int down_sample_factor_;
  double a_frontier_explored_rate_at_least_, resolution_;
  int min_visib_num_, candidate_rnum_;

  // Utils
  shared_ptr<EDTEnvironment> edt_env_;
  unique_ptr<RayCaster> raycaster_;
};

}  // namespace fast_planner
#endif
