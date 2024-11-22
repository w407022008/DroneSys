#include <active_perception/frontier_finder.h>
#include <plan_env/sdf_map.h>
#include <plan_env/raycast.h>
// #include <path_searching/astar2.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <plan_env/edt_environment.h>
#include <active_perception/perception_utils.h>
#include <active_perception/graph_node.h>  //include algorithm, which contains reverse operation

// use PCL region growing segmentation
// #include <pcl/point_types.h>
// #include <pcl/search/search.h>
// #include <pcl/search/kdtree.h>
// #include <pcl/features/normal_3d.h>
// #include <pcl/segmentation/region_growing.h>
#include <pcl/filters/voxel_grid.h>

#include <Eigen/Eigenvalues>

// #define DEBUG

namespace fast_planner {
void FrontierFinder::wrapYaw(double& yaw) {
  while (yaw < -M_PI)
    yaw += 2 * M_PI;
  while (yaw > M_PI)
    yaw -= 2 * M_PI;
}

FrontierFinder::FrontierFinder(const EDTEnvironment::Ptr& edt, ros::NodeHandle& nh) {
  this->edt_env_ = edt;
  int voxel_num = edt->sdf_map_->getVoxelNum();
  cell_is_on_frontier_ = vector<char>(voxel_num, 0);
  fill(cell_is_on_frontier_.begin(), cell_is_on_frontier_.end(), 0);

  nh.param("frontier/cluster_min", cluster_min_, -1);
  nh.param("frontier/cluster_size_xy", cluster_size_xy_, -1.0);// 2
  nh.param("frontier/cluster_size_z", cluster_size_z_, -1.0);
  nh.param("frontier/covered_checking_forced", covered_checking_forced_, true);
  // nh.param("frontier/min_candidate_dist", min_candidate_dist_, -1.0);
  nh.param("frontier/near_unknow_clearance", near_unknow_clearance_, -1.0);
  nh.param("frontier/candidate_dphi", candidate_dphi_, -1.0);
  nh.param("frontier/candidate_rmax", candidate_rmax_, -1.0);
  nh.param("frontier/candidate_rmin", candidate_rmin_, -1.0);
  nh.param("frontier/candidate_rnum", candidate_rnum_, -1);
  nh.param("frontier/down_sample_factor", down_sample_factor_, -1);
  nh.param("frontier/min_visib_num", min_visib_num_, -1);
  nh.param("frontier/a_frontier_explored_rate_at_least", a_frontier_explored_rate_at_least_, -1.0);

  raycaster_.reset(new RayCaster);
  resolution_ = edt_env_->sdf_map_->getResolution();
  Eigen::Vector3d origin, size;
  edt_env_->sdf_map_->getFullMap(origin, size);
  raycaster_->setParams(resolution_, origin);

  percep_utils_.reset(new PerceptionUtils(nh));
}

FrontierFinder::~FrontierFinder() {
}

// Step 1: update frontiers
// =========================================
// Main programme to remove observed frontiers and to search new frontiers
void FrontierFinder::updateFrontiers(bool inLocalSpace) {
  tmp_frontiers_.clear();

  // Bounding box of local updated region
  Vector3d update_min, update_max;
  edt_env_->sdf_map_->getUpdatedBox(update_min, update_max);

  // Removed changed frontiers in updated map
  auto resetFlag = [&](list<Frontier>::iterator& iter, list<Frontier>& frontiers) {
    Eigen::Vector3i idx;
    for (auto cell : iter->cells_) {
      edt_env_->sdf_map_->posToIndex(cell, idx);
      cell_is_on_frontier_[Index3DToAddress1D(idx)] = 0;
    }
    iter = frontiers.erase(iter);
  };

  std::cout << "[frontierFind]:Before remove: " << frontiers_.size() << std::endl;
  // removed_ids_.clear();
  int rmv_idx = 0;
#ifdef DEBUG
  std::cout<<"[DEBUG]: frontiers_ removed ids: "<<std::endl;
#endif
  if(!covered_checking_forced_){
    // code below works since isThereAFrontierCovered() triggers to replan 
    // and then haveOverlap() be true
    // if replan_thresh2_ enough long, covered frontiers may be missed
    for (auto iter = frontiers_.begin(); iter != frontiers_.end();) {
      if (haveOverlap(iter->box_min_, iter->box_max_, update_min, update_max) &&
          isFrontierChanged(*iter)) {
        resetFlag(iter, frontiers_);
        removed_ids_.push_back(rmv_idx);
      } else {
        ++rmv_idx;
        ++iter;
      }
    }
    for (auto iter = dormant_frontiers_.begin(); iter != dormant_frontiers_.end();) {
      if (haveOverlap(iter->box_min_, iter->box_max_, update_min, update_max) &&
          isFrontierChanged(*iter))
        resetFlag(iter, dormant_frontiers_);
      else
        ++iter;
    }
  }else{
    for (auto iter = frontiers_.begin(); iter != frontiers_.end();) {
      if (isFrontierAlmostFullyCovered(*iter)) {
        resetFlag(iter, frontiers_);
        removed_ids_.push_back(rmv_idx);
      } else {
        ++rmv_idx;
        ++iter;
      }
    }
    for (auto iter = dormant_frontiers_.begin(); iter != dormant_frontiers_.end();) {
      if (isFrontierAlmostFullyCovered(*iter))
        resetFlag(iter, dormant_frontiers_);
      else
        ++iter;
    }
  }

  std::cout << "[frontierFind]:After remove: " << frontiers_.size() << std::endl;

  // Search new frontier cells within
  // the slightly inflated local update space 
  // or the whole exploration space
  Vector3d search_min, search_max;
  if(inLocalSpace){
    search_min = update_min - Vector3d(1, 1, 0.5);
    search_max = update_max + Vector3d(1, 1, 0.5);
    Vector3d box_min, box_max;
    edt_env_->sdf_map_->getFullBox(box_min, box_max);
    for (int k = 0; k < 3; ++k) {
      search_min[k] = max(search_min[k], box_min[k]);
      search_max[k] = min(search_max[k], box_max[k]);
    }
  }else{
    edt_env_->sdf_map_->getFullBox(search_min, search_max);
  }

  Eigen::Vector3i min_id, max_id;
  edt_env_->sdf_map_->posToIndex(search_min, min_id);
  edt_env_->sdf_map_->posToIndex(search_max, max_id);

  for (int x = min_id(0); x <= max_id(0); ++x)
    for (int y = min_id(1); y <= max_id(1); ++y)
      for (int z = min_id(2); z <= max_id(2); ++z) {
        // Scanning the updated region to find seeds of frontiers
        Eigen::Vector3i cur(x, y, z);
        if(!inmap(cur)) continue;
        if (cell_is_on_frontier_[Index3DToAddress1D(cur)] == 0 && knownfree(cur) && hasUnknownNeighbor(cur)) {
          // Expand from the seed cell to find a complete frontier cluster
          expandFrontier(cur);
        }
      }
  std::cout << "[frontierFind]:expanded " << std::endl;
  splitLargeFrontiers(tmp_frontiers_);
  std::cout << "[frontierFind]:splitted " << std::endl;
}

void FrontierFinder::expandFrontier(
    const Eigen::Vector3i& first /* , const int& depth, const int& parent_id */) {

  // Data for clustering
  queue<Eigen::Vector3i> cell_queue;
  vector<Eigen::Vector3d> expanded;
  Vector3d pos;

  edt_env_->sdf_map_->indexToPos(first, pos);
  expanded.push_back(pos);
  cell_queue.push(first);
  cell_is_on_frontier_[Index3DToAddress1D(first)] = 1;

  // Search frontier cluster based on region growing (distance clustering)
  while (!cell_queue.empty()) {
    auto cur = cell_queue.front();
    cell_queue.pop();
    auto nbrs = allNeighbors(cur);
    for (auto nbr : nbrs) {
      // Qualified cell should be inside bounding box and frontier cell not clustered
      if(!inmap(nbr)) continue;
      int adr = Index3DToAddress1D(nbr);
      if (cell_is_on_frontier_[adr] == 1 || !edt_env_->sdf_map_->isInBox(nbr) ||
          !(knownfree(nbr) && hasUnknownNeighbor(nbr)))
        continue;

      edt_env_->sdf_map_->indexToPos(nbr, pos);
      if (pos[2] < 0.4) continue;  // Remove noise close to ground
      expanded.push_back(pos);
      cell_queue.push(nbr);
      cell_is_on_frontier_[adr] = 1;
    }
  }
  if (expanded.size() >= cluster_min_) {
    // Compute detailed info
    Frontier frontier;
    frontier.cells_ = expanded;
    computeCellsAverage_CellsBox_FilteredCells_Of(frontier);
    tmp_frontiers_.push_back(frontier);
  }
}

void FrontierFinder::splitLargeFrontiers(list<Frontier>& frontiers) {
  list<Frontier> splits, tmps;
  for (auto it = frontiers.begin(); it != frontiers.end(); ++it) {
    // Check if each frontier needs to be split horizontally
    if (splitHorizontally(*it, splits)) {
      tmps.insert(tmps.end(), splits.begin(), splits.end());
      splits.clear();
    } else
      tmps.push_back(*it);
  }
  frontiers = tmps;
}

bool FrontierFinder::splitHorizontally(const Frontier& frontier, list<Frontier>& splits) {
  // Split a frontier into small piece if it is too large
  auto mean = frontier.average_.head<2>();
  bool need_split = false;
  for (auto cell : frontier.filtered_cells_) {
    if ((cell.head<2>() - mean).norm() > cluster_size_xy_) {
      need_split = true;
      break;
    }
  }
  if (!need_split) return false;

  // Compute principal component
  // Covariance matrix of cells
  Eigen::Matrix2d cov;
  cov.setZero();
  for (auto cell : frontier.filtered_cells_) {
    Eigen::Vector2d diff = cell.head<2>() - mean;
    cov += diff * diff.transpose();
  }
  cov /= double(frontier.filtered_cells_.size());

  // Find eigenvector corresponds to maximal eigenvector
  Eigen::EigenSolver<Eigen::Matrix2d> es(cov);
  auto values = es.eigenvalues().real();
  auto vectors = es.eigenvectors().real();
  int max_idx;
  double max_eigenvalue = -1000000;
  for (int i = 0; i < values.rows(); ++i) {
    if (values[i] > max_eigenvalue) {
      max_idx = i;
      max_eigenvalue = values[i];
    }
  }
  Eigen::Vector2d first_principle_axis = vectors.col(max_idx);

#ifdef DEBUG
  std::cout << "[DEBUG]:frontier center xy: " << mean.transpose() << ", first principle axis: " << first_principle_axis.transpose() << std::endl;
#endif

  // Split the frontier into two groups vers/inverse the first principle axis
  Frontier ftr1, ftr2;
  for (auto cell : frontier.cells_) {
    if ((cell.head<2>() - mean).dot(first_principle_axis) >= 0)
      ftr1.cells_.push_back(cell);
    else
      ftr2.cells_.push_back(cell);
  }
  computeCellsAverage_CellsBox_FilteredCells_Of(ftr1);
  computeCellsAverage_CellsBox_FilteredCells_Of(ftr2);

  // Recursive call to split frontier that is still too large
  list<Frontier> splits2;
  if (splitHorizontally(ftr1, splits2)) {
    splits.insert(splits.end(), splits2.begin(), splits2.end());
    splits2.clear();
  } else
    splits.push_back(ftr1);

  if (splitHorizontally(ftr2, splits2))
    splits.insert(splits.end(), splits2.begin(), splits2.end());
  else
    splits.push_back(ftr2);

  return true;
}

void FrontierFinder::mergeFrontiers(Frontier& ftr1, const Frontier& ftr2) {
  // Merge ftr2 into ftr1
  ftr1.average_ =
      (ftr1.average_ * double(ftr1.cells_.size()) + ftr2.average_ * double(ftr2.cells_.size())) /
      (double(ftr1.cells_.size() + ftr2.cells_.size()));
  ftr1.cells_.insert(ftr1.cells_.end(), ftr2.cells_.begin(), ftr2.cells_.end());
  computeCellsAverage_CellsBox_FilteredCells_Of(ftr1);
}

bool FrontierFinder::canBeMerged(const Frontier& ftr1, const Frontier& ftr2) {
  Vector3d merged_avg =
      (ftr1.average_ * double(ftr1.cells_.size()) + ftr2.average_ * double(ftr2.cells_.size())) /
      (double(ftr1.cells_.size() + ftr2.cells_.size()));
  // Check if it can merge two frontier without exceeding size limit
  for (auto c1 : ftr1.cells_) {
    auto diff = c1 - merged_avg;
    if (diff.head<2>().norm() > cluster_size_xy_ || diff[2] > cluster_size_z_) return false;
  }
  for (auto c2 : ftr2.cells_) {
    auto diff = c2 - merged_avg;
    if (diff.head<2>().norm() > cluster_size_xy_ || diff[2] > cluster_size_z_) return false;
  }
  return true;
}

void FrontierFinder::computeCellsAverage_CellsBox_FilteredCells_Of(Frontier& frt) {
  // Compute average position and bounding box of cluster
  frt.average_.setZero();
  frt.box_max_ = frt.cells_.front();
  frt.box_min_ = frt.cells_.front();
  for (auto cell : frt.cells_) {
    frt.average_ += cell;
    for (int i = 0; i < 3; ++i) {
      frt.box_min_[i] = min(frt.box_min_[i], cell[i]);
      frt.box_max_[i] = max(frt.box_max_[i], cell[i]);
    }
  }
  frt.average_ /= double(frt.cells_.size());

  // Compute downsampled cluster
  downsample(frt.cells_, frt.filtered_cells_);
}

void FrontierFinder::downsample(
    const vector<Eigen::Vector3d>& cluster_in, vector<Eigen::Vector3d>& cluster_out) {
  // downsamping cluster
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloudf(new pcl::PointCloud<pcl::PointXYZ>);
  for (auto cell : cluster_in)
    cloud->points.emplace_back(cell[0], cell[1], cell[2]);

  const double leaf_size = edt_env_->sdf_map_->getResolution() * down_sample_factor_;
  pcl::VoxelGrid<pcl::PointXYZ> sor;
  sor.setInputCloud(cloud);
  sor.setLeafSize(leaf_size, leaf_size, leaf_size);
  sor.filter(*cloudf);

  cluster_out.clear();
  for (auto pt : cloudf->points)
    cluster_out.emplace_back(pt.x, pt.y, pt.z);
}

// Step 2: search corresponding viewpoints of new frontiers
// =========================================
//  search new viewpoints, after searching new frontier(tmp_frontiers_) cells
void FrontierFinder::searchViewpointsOfNewFrontiers() {
  first_new_ftr_ = frontiers_.end();
  last_frontier_size = frontiers_.size();
#ifdef DEBUG
  std::cout<<"[DEBUG]:old frontier num: " << frontiers_.size()<<std::endl;;
#endif
  int new_num = 0;
  int new_dormant_num = 0;
  // Try find viewpoints for each cluster and categorize them according to viewpoint number
  for (auto& tmp_ftr : tmp_frontiers_) {
    // Search viewpoints around frontier
    sampleViewpoints(tmp_ftr);
    if (!tmp_ftr.viewpoints_.empty()) {
      ++new_num;
      list<Frontier>::iterator inserted = frontiers_.insert(frontiers_.end(), tmp_ftr);
      // Sort the viewpoints by coverage fraction, best view in front
      sort(
          inserted->viewpoints_.begin(), inserted->viewpoints_.end(),
          [](const Viewpoint& v1, const Viewpoint& v2) { return v1.visib_num_ > v2.visib_num_; });
      if (first_new_ftr_ == frontiers_.end()) first_new_ftr_ = inserted;
    } else {
      // Find no viewpoint, move cluster to dormant list
      dormant_frontiers_.push_back(tmp_ftr);
      ++new_dormant_num;
    }
  }
  cur_frontier_size = frontiers_.size();
  // Reset indices of frontiers
  int idx = 0;
#ifdef DEBUG
  std::cout<<"[DEBUG]:frontier id: ";
#endif
  for (auto& ft : frontiers_) {
    ft.id_ = idx++;
#ifdef DEBUG
    std::cout << ft.id_ << ", ";
#endif
  }
#ifdef DEBUG
  std::cout<<std::endl;
  std::cout <<"[DEBUG]:new one from: "<< first_new_ftr_->id_ << std::endl;
#endif
  std::cout << "[frontierFind]:new frontiers: " << new_num << 
                ", new dormant frontiers: " << new_dormant_num << std::endl;
  std::cout << "[frontierFind]:total to visit: " << frontiers_.size() << 
                ", dormant: " << dormant_frontiers_.size() << std::endl;
}

// Sample viewpoints around frontier's average position, check coverage to the frontier cells
void FrontierFinder::sampleViewpoints(Frontier& frontier) {
  // Evaluate sample viewpoints on circles, find ones that cover most cells
  for (double rc = candidate_rmin_, dr = (candidate_rmax_ - candidate_rmin_) / candidate_rnum_;
       rc <= candidate_rmax_ + 1e-3; rc += dr)
    for (double phi = -M_PI; phi < M_PI; phi += candidate_dphi_) {
      const Vector3d sample_pos = frontier.average_ + rc * Vector3d(cos(phi), sin(phi), 0);

      // Qualified viewpoint is in bounding box and in safe region
      if (!edt_env_->sdf_map_->isInBox(sample_pos) ||
          edt_env_->sdf_map_->getInflateOccupancy(sample_pos) == 1 || hasNearUnknown(sample_pos))
        continue;

      // Compute average yaw
      auto& cells = frontier.filtered_cells_;
      Eigen::Vector3d ref_dir = (cells.front() - sample_pos).normalized();
      double avg_yaw = 0.0;
      for (int i = 1; i < cells.size(); ++i) {
        Eigen::Vector3d dir = (cells[i] - sample_pos).normalized();
        double yaw = acos(dir.dot(ref_dir));
        if (ref_dir.cross(dir)[2] < 0) yaw = -yaw;
        avg_yaw += yaw;
      }
      avg_yaw = avg_yaw / cells.size() + atan2(ref_dir[1], ref_dir[0]);
      wrapYaw(avg_yaw);
      // Compute the fraction of covered and visible cells
      int visib_num = countVisibleCells(sample_pos, avg_yaw, cells);
      if (visib_num > min_visib_num_) {
        Viewpoint vp = { sample_pos, avg_yaw, visib_num };
        frontier.viewpoints_.push_back(vp);
        // int gain = findMaxGainYaw(sample_pos, frontier, sample_yaw);
      }
      // }
    }
}

// How many frontier cells are visible from a viewpoint
int FrontierFinder::countVisibleCells(
    const Eigen::Vector3d& pos, const double& yaw, const vector<Eigen::Vector3d>& cluster) {
  percep_utils_->setPose(pos, yaw);
  int visib_num = 0;
  Eigen::Vector3i idx;
  for (auto cell : cluster) {
    // Check if frontier cell is inside FOV
    if (!percep_utils_->insideFOV(cell)) continue;

    // Check if frontier cell is visible (not occulded by obstacles)
    raycaster_->setInput(cell, pos);
    bool visib = true;
    while (raycaster_->nextId(idx)) {
      if (edt_env_->sdf_map_->getInflateOccupancy(idx) == 1 ||
          edt_env_->sdf_map_->getOccupancy(idx) == SDFMap::UNKNOWN) {
        visib = false;
        break;
      }
    }
    if (visib) visib_num += 1;
  }
  return visib_num;
}

// Step 3: couple all frontiers to find a path to each other and then compute co-cost
// =========================================
// path a frontier to a frontier, and compute their co- cost matrix
void FrontierFinder::updatePathAndCostAmongTopFrontierViewpoints() {
#ifdef DEBUG
  std::cout << "[DEBUG]:cost mat size before remove: " << std::endl;
  std::cout << "frontiers: " << frontiers_.size() << ", costs mtx: " 
      << frontiers_.back().costs_.size() << ", path: " << frontiers_.back().paths_.size() << std::endl;
  // for (auto frt : frontiers_)
  //   std::cout << "(" <<frt.id_<< "," << frt.costs_.size() << "," << frt.paths_.size() << "), ";
  // std::cout << "" << std::endl;
#endif

  int removed_size = 0;
  if (!removed_ids_.empty()) {
    removed_size = removed_ids_.size();
#ifdef DEBUG
    std::cout<<"[DEBUG]:removed: "<< removed_size <<std::endl;
    for (int i = 0; i < removed_size; ++i)
      std::cout<<removed_ids_[i]<<" ";
    std::cout<<std::endl;
    std::cout << "[DEBUG]:cost mat size remove: " << std::endl;
#endif
    // Delete path and cost for removed clusters
    if(last_frontier_size == cur_frontier_size) first_new_ftr_ = frontiers_.end();
    for (auto it = frontiers_.begin(); it != first_new_ftr_; ++it) {
#ifdef DEBUG
      std::cout<<".";
#endif
      auto cost_iter = it->costs_.begin();
      auto path_iter = it->paths_.begin();
      int iter_idx = 0;
      for (int i = 0; i < removed_size; ++i) {
        // Step iterator to the item to be removed
        while (iter_idx < removed_ids_[i]) {
          ++cost_iter;
          ++path_iter;
          ++iter_idx;
        }
        cost_iter = it->costs_.erase(cost_iter);
        path_iter = it->paths_.erase(path_iter);
      }
#ifdef DEBUG
      // std::cout << "(" <<it->id_<< "," << it->costs_.size() << "," << it->paths_.size() << "), ";
#endif
    }
    removed_ids_.clear();
#ifdef DEBUG
    // std::cout << std::endl;
  std::cout << "frontiers: " << frontiers_.size() << ", costs mtx: " 
      << frontiers_.back().costs_.size() << ", path: " << frontiers_.back().paths_.size() << std::endl;
#endif
  }
#ifdef DEBUG
  else{
    std::cout<<"null removed ids"<<std::endl;
  }
#endif

  auto updateCost = [](const list<Frontier>::iterator& it1, const list<Frontier>::iterator& it2) {
#ifdef DEBUG
    // std::cout << "(" << it1->id_ << "," << it2->id_ << "), ";
#endif
    // Search path from old cluster's top viewpoint to new cluster'
    Viewpoint& vui = it1->viewpoints_.front();
    Viewpoint& vuj = it2->viewpoints_.front();
    vector<Vector3d> path_ij;
    double cost_ij = ViewNode::computeCost(
        vui.pos_, vuj.pos_, vui.yaw_, vuj.yaw_, Vector3d(0, 0, 0), 0, path_ij);// obtain cost and path
    // Insert item for both old and new clusters
    it1->costs_.push_back(cost_ij);
    it1->paths_.push_back(path_ij);
    reverse(path_ij.begin(), path_ij.end());
    it2->costs_.push_back(cost_ij);
    it2->paths_.push_back(path_ij);
  };

#ifdef DEBUG
  std::cout << "[DEBUG]:cost mat add: "<< std::endl;
#endif
  // Compute path and cost between old and new clusters
  if(last_frontier_size == cur_frontier_size) first_new_ftr_ = frontiers_.end();
  for (auto it1 = frontiers_.begin(); it1 != first_new_ftr_; ++it1)
    for (auto it2 = first_new_ftr_; it2 != frontiers_.end(); ++it2)
      updateCost(it1, it2);

  // Compute path and cost between new clusters
  for (auto it1 = first_new_ftr_; it1 != frontiers_.end(); ++it1)
    for (auto it2 = it1; it2 != frontiers_.end(); ++it2) {
      if (it1 == it2) {
#ifdef DEBUG
        // std::cout << "(" << it1->id_ << "," << it2->id_ << "), ";
#endif
        it1->costs_.push_back(0);
        it1->paths_.push_back({});
      } else
        updateCost(it1, it2);
    }
#ifdef DEBUG
  std::cout << "" << std::endl;
  std::cout << "[DEBUG]:cost mat size final: " << std::endl;
  std::cout << "frontiers: " << frontiers_.size() << ", costs mtx: " 
      << frontiers_.back().costs_.size() << ", path: " << frontiers_.back().paths_.size() << std::endl;
  for (auto frt : frontiers_)
    std::cout << "(" <<frt.id_<< "," << frt.costs_.size() << "," << frt.paths_.size() << "), ";
  std::cout << "" << std::endl;
#endif
}

// compute full co- cost matrix among top frontier viewpoints
void FrontierFinder::computeFullCostMatrixAmongTopFrontierViewpointsFrom(
    const Vector3d& cur_pos, const Vector3d& cur_vel, const Vector3d cur_yaw,
    Eigen::MatrixXd& mat) {
  // list<Frontier> frontier_copy;
  // for(const auto& elem:frontiers_){
  //   frontier_copy.push_back(elem);
  // }  
#ifdef DEBUG
    std::cout << "start cost matrix computing " << std::endl;
#endif
  if (false) {
    // Use symmetric TSP formulation
    int dim = frontiers_.size() + 2;
    mat.resize(dim, dim);  // current pose (0), sites, and virtual depot finally

    int i = 1, j = 1;
    for (auto frt : frontiers_) {
      for (auto cs : frt.costs_)
        mat(i, j++) = cs;
      ++i;
      j = 1;
    }

    // Costs from current pose to sites
    for (auto frt : frontiers_) {
      Viewpoint vj = frt.viewpoints_.front();
      vector<Vector3d> path;
      mat(0, j) = mat(j, 0) =
          ViewNode::computeCost(cur_pos, vj.pos_, cur_yaw[0], vj.yaw_, cur_vel, cur_yaw[1], path);
      ++j;
    }
    // Costs from depot to sites, the same large vaule
    for (j = 1; j < dim - 1; ++j) {
      mat(dim - 1, j) = mat(j, dim - 1) = 100;
    }
    // Zero cost to depot to ensure connection
    mat(0, dim - 1) = mat(dim - 1, 0) = -10000;

  } else {
    // Use Asymmetric TSP
    int dimen = frontiers_.size();
    mat.resize(dimen + 1, dimen + 1);

    // Fill block for clusters
    int i = 1, j = 1;
    for (auto frt : frontiers_) {
      for (auto cs : frt.costs_) {
        mat(i, j++) = cs;
      }
      ++i;
      j = 1;
    }

    // Fill block from current state to clusters
    mat.leftCols<1>().setZero();
    for (auto frt : frontiers_) {
      Viewpoint vj = frt.viewpoints_.front();
      vector<Vector3d> path;
      double cost = ViewNode::computeCost(cur_pos, vj.pos_, cur_yaw[0], vj.yaw_, cur_vel, cur_yaw[1], path);;
      mat(0, j++) = cost;
#ifdef DEBUG
    std::cout << cost << ", ";
#endif
    }
#ifdef DEBUG
    std::cout<<std::endl;
#endif
  }
#ifdef DEBUG
    std::cout << "tsp cost matrix size: " << mat.rows() << std::endl;
#endif
}

// Helper functions
// =========================================
bool FrontierFinder::isThereAFrontierCovered() {
  // get the last pcl frame updating box
  Vector3d update_min, update_max;
  edt_env_->sdf_map_->getUpdatedBox(update_min, update_max);

  for (auto frt : frontiers_) {
    if (!haveOverlap(frt.box_min_, frt.box_max_, update_min, update_max)) continue;
    if(covered_checking_forced_)
      if(isFrontierAlmostFullyCovered(frt)) return true;
    else 
      if(isFrontierChanged(frt)) return true;
  }

  // for (auto frt : dormant_frontiers_) {
  //   if (!haveOverlap(frt.box_min_, frt.box_max_, update_min, update_max)) continue;
  //   if(isFrontierChanged(frt)) return true;
  // }

  return false;
}

bool FrontierFinder::isFrontierAlmostFullyCovered(const Frontier& ft) {
  int change_num = 0;
  const int change_thresh = a_frontier_explored_rate_at_least_ * ft.cells_.size();
  for (auto cell : ft.cells_) {
    Eigen::Vector3i idx;
    edt_env_->sdf_map_->posToIndex(cell, idx);
    if (!(knownfree(idx) && hasUnknownNeighbor(idx)) && ++change_num >= change_thresh){
#ifdef DEBUG
      std::cout<< "[DEBUG]: almost fully covered frontier id:"<<ft.id_<<std::endl;
#endif
      return true;
    }
  }
  return false;
}

bool FrontierFinder::isFrontierChanged(const Frontier& ft) {
  for (auto cell : ft.cells_) {
    Eigen::Vector3i idx;
    edt_env_->sdf_map_->posToIndex(cell, idx);
    if (!(knownfree(idx) && hasUnknownNeighbor(idx))) {
#ifdef DEBUG
      std::cout<< "[DEBUG]: changing frontier id:"<<ft.id_<<std::endl;
#endif
      return true;
    }
  }
  return false;
}

bool FrontierFinder::haveOverlap(
    const Vector3d& min1, const Vector3d& max1, const Vector3d& min2, const Vector3d& max2) {
  // Check if two box have overlap part
  Vector3d bmin, bmax;
  for (int i = 0; i < 3; ++i) {
    bmin[i] = max(min1[i], min2[i]);
    bmax[i] = min(max1[i], max2[i]);
    if (bmin[i] > bmax[i] + 1e-3) return false;
  }
  return true;
}

void FrontierFinder::getTopViewpointOfEachFrontier(
    vector<Eigen::Vector3d>& points, vector<double>& yaws, vector<Eigen::Vector3d>& averages) {
  points.clear();
  yaws.clear();
  averages.clear();
  for (auto frontier : frontiers_) {
    // All viewpoints are very close, just use the first one (with highest coverage, has been sorted).
    auto viewpoint = frontier.viewpoints_.front();
    points.push_back(viewpoint.pos_);
    yaws.push_back(viewpoint.yaw_);
    averages.push_back(frontier.average_);
  }
}

void FrontierFinder::getFarEnoughTopViewpointOfEachFrontierFrom(
    const Vector3d& cur_pos, vector<Eigen::Vector3d>& points, vector<double>& yaws,
    vector<Eigen::Vector3d>& averages, double min_candidate_dist_) {
  points.clear();
  yaws.clear();
  averages.clear();
  for (auto frontier : frontiers_) {
    bool all_viewpoints_too_close = true;
    for (auto viewpoint : frontier.viewpoints_) {
      // Retrieve the first viewpoint that is far enough and has highest coverage
      if ((viewpoint.pos_ - cur_pos).norm() < min_candidate_dist_) continue;// frontier/min_candidate_dist=0.75
      points.push_back(viewpoint.pos_);
      yaws.push_back(viewpoint.yaw_);
      averages.push_back(frontier.average_);
      all_viewpoints_too_close = false;
      break;
    }
    if (all_viewpoints_too_close) {
      // All viewpoints are very close, just use the first one (with highest coverage, has been sorted).
      auto viewpoint = frontier.viewpoints_.front();
      points.push_back(viewpoint.pos_);
      yaws.push_back(viewpoint.yaw_);
      averages.push_back(frontier.average_);
    }
  }
}

void FrontierFinder::getConditionalTopViewpointsFrom_OF(
    const Vector3d& cur_pos, const vector<int>& ids, const int& view_num, const double& max_decay,
    vector<vector<Eigen::Vector3d>>& points, vector<vector<double>>& yaws, double min_candidate_dist_) {
  points.clear();
  yaws.clear();
  for (auto id : ids) {
    // Scan all frontiers to find one with the same id
    for (auto frontier : frontiers_) {
      if (frontier.id_ == id) {
        // Get several(view_num) top viewpoints(visib_thresh) that are far enough(min_candidate_dist_)
        vector<Eigen::Vector3d> pts;
        vector<double> ys;
        int visib_thresh = frontier.viewpoints_.front().visib_num_ * max_decay;
        for (auto viewpoint : frontier.viewpoints_) {
          if (pts.size() >= view_num || viewpoint.visib_num_ <= visib_thresh) break;
          if ((viewpoint.pos_ - cur_pos).norm() < min_candidate_dist_) continue;
          pts.push_back(viewpoint.pos_);
          ys.push_back(viewpoint.yaw_);
        }
        if (pts.empty()) {
          // All viewpoints are very close, ignore the distance limit
          for (auto viewpoint : frontier.viewpoints_) {
            if (pts.size() >= view_num || viewpoint.visib_num_ <= visib_thresh) break;
            pts.push_back(viewpoint.pos_);
            ys.push_back(viewpoint.yaw_);
          }
        }
        points.push_back(pts);
        yaws.push_back(ys);
      }
    }
  }
}

void FrontierFinder::getTopViewpointsOf_From(
    const Vector3d& cur_pos, const vector<int>& ids, const int& view_num, const double& max_decay,
    vector<vector<Eigen::Vector3d>>& points, vector<vector<double>>& yaws, vector<Eigen::Vector3d>& averages, 
    double min_candidate_dist_) {
  assert(points.empty() && yaws.empty() && averages.empty());
  int far_id;
  double far_dist = 0.0;
  for (auto id : ids) {
    // Scan all frontiers to find one with the same id
    for (auto frontier : frontiers_) {
      if (frontier.id_ == id) {
        double dist = (frontier.viewpoints_[0].pos_-cur_pos).norm();
        if(dist<min_candidate_dist_) {
          if(far_dist<dist){
            far_id = id;
            far_dist = dist;
          }
          continue;
        }
        // Get several(view_num) top viewpoints(visib_thresh)
        vector<Eigen::Vector3d> pts;
        vector<double> ys;
        int visib_thresh = frontier.viewpoints_.front().visib_num_ * max_decay;
        for (auto viewpoint : frontier.viewpoints_) {
          if (pts.size() >= view_num || viewpoint.visib_num_ <= visib_thresh) break;
          pts.push_back(viewpoint.pos_);
          ys.push_back(viewpoint.yaw_);
        }
        points.push_back(pts);
        yaws.push_back(ys);
        averages.push_back(frontier.average_);
      }
    }
  }
  if(points.empty()){
    for (auto frontier : frontiers_) {
      if (frontier.id_ == far_id) {
        // Get several(view_num) top viewpoints(visib_thresh)
        vector<Eigen::Vector3d> pts;
        vector<double> ys;
        int visib_thresh = frontier.viewpoints_.front().visib_num_ * max_decay;
        for (auto viewpoint : frontier.viewpoints_) {
          pts.push_back(viewpoint.pos_);
          ys.push_back(viewpoint.yaw_);
        }
        points.push_back(pts);
        yaws.push_back(ys);
        averages.push_back(frontier.average_);
      }
    }
  }
}

void FrontierFinder::getCellsOfEachFrontier(vector<vector<Eigen::Vector3d>>& clusters_frontier,
                                            vector<vector<Eigen::Vector3d>>& clusters_dormant_frontier) {
  assert(clusters_frontier.empty() && clusters_dormant_frontier.empty() && "getCellsOfEachFrontier input not empty");
  std::cout<<"getting frontiers cells"<<std::endl;
  for (auto frontier : frontiers_)
    clusters_frontier.push_back(frontier.cells_);
  // clusters_frontier.push_back(frontier.filtered_cells_);
  std::cout<<"getting dormant frontiers cells"<<std::endl;
  for (auto frontier : dormant_frontiers_)
    clusters_dormant_frontier.push_back(frontier.cells_);
}

void FrontierFinder::getBoundingBoxOfEachFrontier(vector<pair<Eigen::Vector3d, Eigen::Vector3d>>& boxes) {
  assert(boxes.empty());
  for (auto frontier : frontiers_) {
    Vector3d center = (frontier.box_max_ + frontier.box_min_) * 0.5;
    Vector3d side_length = frontier.box_max_ - frontier.box_min_;
    boxes.push_back(make_pair(center, side_length));
  }
}

void FrontierFinder::getPathAllFrontiersFrom_Along(
    const Vector3d& pos, const vector<int>& frontier_ids, vector<Vector3d>& path) {
  // Make an frontier_indexer to access the frontier list easier
  vector<list<Frontier>::iterator> frontier_indexer;
  for (auto it = frontiers_.begin(); it != frontiers_.end(); ++it)
    frontier_indexer.push_back(it);

  // Compute the path from current pos to the first frontier
  vector<Vector3d> segment;
  ViewNode::searchPath(pos, frontier_indexer[frontier_ids[0]]->viewpoints_.front().pos_, segment);
  path.insert(path.end(), segment.begin(), segment.end());

  // Get paths of tour passing all clusters
  for (int i = 0; i < frontier_ids.size() - 1; ++i) {
    // Move to path to next cluster
    auto path_iter = frontier_indexer[frontier_ids[i]]->paths_.begin();
    int next_idx = frontier_ids[i + 1];
    for (int j = 0; j < next_idx; ++j)
      ++path_iter;
    if(!path_iter->empty())
      path.insert(path.end(), path_iter->begin(), path_iter->end());
  }
}

int FrontierFinder::countFrontierCells(){
  int cnt=0;
  for(auto elem:cell_is_on_frontier_){
    if(elem == 1){
      ++cnt;
    }
  }
  return cnt;
}

bool FrontierFinder::hasNearUnknown(const Eigen::Vector3d& pos) {
  const int vox_num = floor(near_unknow_clearance_ / resolution_);
  for (int x = -vox_num; x <= vox_num; ++x)
    for (int y = -vox_num; y <= vox_num; ++y)
      for (int z = -1; z <= 1; ++z) {
        Eigen::Vector3d vox;
        vox << pos[0] + x * resolution_, pos[1] + y * resolution_, pos[2] + z * resolution_;
        if (edt_env_->sdf_map_->getOccupancy(vox) == SDFMap::UNKNOWN) return true;
      }
  return false;
}

Eigen::Vector3i FrontierFinder::searchClearVoxel(const Eigen::Vector3i& pt) {
  queue<Eigen::Vector3i> init_que;
  vector<Eigen::Vector3i> nbrs;
  Eigen::Vector3i cur, start_idx;
  init_que.push(pt);
  // visited_flag_[Index3DToAddress1D(pt)] = 1;

  while (!init_que.empty()) {
    cur = init_que.front();
    init_que.pop();
    if (knownfree(cur)) {
      start_idx = cur;
      break;
    }

    nbrs = sixNeighbors(cur);
    for (auto nbr : nbrs) {
      int adr = Index3DToAddress1D(nbr);
      // if (visited_flag_[adr] == 0)
      // {
      //   init_que.push(nbr);
      //   visited_flag_[adr] = 1;
      // }
    }
  }
  return start_idx;
}

inline vector<Eigen::Vector3i> FrontierFinder::sixNeighbors(const Eigen::Vector3i& voxel) {
  vector<Eigen::Vector3i> neighbors(6);
  Eigen::Vector3i tmp;

  tmp = voxel - Eigen::Vector3i(1, 0, 0);
  neighbors[0] = tmp;
  tmp = voxel + Eigen::Vector3i(1, 0, 0);
  neighbors[1] = tmp;
  tmp = voxel - Eigen::Vector3i(0, 1, 0);
  neighbors[2] = tmp;
  tmp = voxel + Eigen::Vector3i(0, 1, 0);
  neighbors[3] = tmp;
  tmp = voxel - Eigen::Vector3i(0, 0, 1);
  neighbors[4] = tmp;
  tmp = voxel + Eigen::Vector3i(0, 0, 1);
  neighbors[5] = tmp;

  return neighbors;
}

inline vector<Eigen::Vector3i> FrontierFinder::tenNeighbors(const Eigen::Vector3i& voxel) {
  vector<Eigen::Vector3i> neighbors(10);
  Eigen::Vector3i tmp;
  int count = 0;

  for (int x = -1; x <= 1; ++x) {
    for (int y = -1; y <= 1; ++y) {
      if (x == 0 && y == 0) continue;
      tmp = voxel + Eigen::Vector3i(x, y, 0);
      neighbors[count++] = tmp;
    }
  }
  neighbors[count++] = tmp - Eigen::Vector3i(0, 0, 1);
  neighbors[count++] = tmp + Eigen::Vector3i(0, 0, 1);
  return neighbors;
}

inline vector<Eigen::Vector3i> FrontierFinder::allNeighbors(const Eigen::Vector3i& voxel) {
  vector<Eigen::Vector3i> neighbors(26);
  Eigen::Vector3i tmp;
  int count = 0;
  for (int x = -1; x <= 1; ++x)
    for (int y = -1; y <= 1; ++y)
      for (int z = -1; z <= 1; ++z) {
        if (x == 0 && y == 0 && z == 0) continue;
        tmp = voxel + Eigen::Vector3i(x, y, z);
        neighbors[count++] = tmp;
      }
  return neighbors;
}

inline bool FrontierFinder::hasUnknownNeighbor(const Eigen::Vector3i& voxel) {
  // At least one neighbor is unknown
  auto nbrs = sixNeighbors(voxel);
  for (auto nbr : nbrs) {
    if (edt_env_->sdf_map_->getOccupancy(nbr) == SDFMap::UNKNOWN) return true;
  }
  return false;
}

inline int FrontierFinder::Index3DToAddress1D(const Eigen::Vector3i& idx) {
  return edt_env_->sdf_map_->toAddress(idx);
}

inline bool FrontierFinder::knownfree(const Eigen::Vector3i& idx) {
  return edt_env_->sdf_map_->getOccupancy(idx) == SDFMap::FREE;
}

inline bool FrontierFinder::inmap(const Eigen::Vector3i& idx) {
  return edt_env_->sdf_map_->isInMap(idx);
}

bool FrontierFinder::isInBoxes(
    const vector<pair<Vector3d, Vector3d>>& boxes, const Eigen::Vector3i& idx) {
  Vector3d pt;
  edt_env_->sdf_map_->indexToPos(idx, pt);
  for (auto box : boxes) {
    // Check if contained by a box
    bool inbox = true;
    for (int i = 0; i < 3; ++i) {
      inbox = inbox && pt[i] > box.first[i] && pt[i] < box.second[i];
      if (!inbox) break;
    }
    if (inbox) return true;
  }
  return false;
}

}  // namespace fast_planner
