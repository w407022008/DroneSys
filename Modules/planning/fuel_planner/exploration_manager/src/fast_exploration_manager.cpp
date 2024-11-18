// #include <fstream>
#include <exploration_manager/fast_exploration_manager.h>
#include <thread>
#include <iostream>
#include <fstream>
#include <lkh_tsp_solver/lkh_interface.h>
#include <active_perception/graph_node.h>
#include <active_perception/graph_search.h>
#include <active_perception/perception_utils.h>
#include <plan_env/raycast.h>
#include <plan_env/sdf_map.h>
#include <plan_env/edt_environment.h>
#include <active_perception/frontier_finder.h>
#include <plan_manage/planner_manager.h>

#include <exploration_manager/expl_data.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <visualization_msgs/Marker.h>

// #define DEBUG

using namespace Eigen;

namespace fast_planner {
// SECTION interfaces for setup and query

FastExplorationManager::FastExplorationManager() {
}

FastExplorationManager::~FastExplorationManager() {
  ViewNode::astar_.reset();
  ViewNode::raycaster_.reset();
  ViewNode::map_.reset();
}

void FastExplorationManager::initialize(ros::NodeHandle& nh) {
  planner_manager_.reset(new FastPlannerManager);
  planner_manager_->initPlanModules(nh);
  edt_environment_ = planner_manager_->edt_environment_;
  sdf_map_ = edt_environment_->sdf_map_;
  frontier_finder_.reset(new FrontierFinder(edt_environment_, nh));
  // view_finder_.reset(new ViewFinder(edt_environment_, nh));

  ed_.reset(new ExplorationData);
  ep_.reset(new ExplorationParam);

  nh.param("exploration/refine_local", ep_->refine_local_, true);
  nh.param("exploration/refined_num", ep_->refined_num_, -1);
  nh.param("exploration/refined_radius", ep_->refined_radius_, -1.0);
  nh.param("exploration/top_view_num", ep_->top_view_num_, -1);
  nh.param("exploration/max_decay", ep_->max_decay_, -1.0);
  nh.param("exploration/tsp_dir", ep_->tsp_dir_, string("null"));
  nh.param("exploration/relax_time", ep_->relax_time_, 1.0);
  nh.param("exploration/min_candidate_dist", ep_->min_candidate_dist_, 1.0);

  nh.param("exploration/vm", ViewNode::vm_, -1.0);// 1*max_vel  3m/s
  nh.param("exploration/am", ViewNode::am_, -1.0);
  nh.param("exploration/yd", ViewNode::yd_, -1.0);
  nh.param("exploration/ydd", ViewNode::ydd_, -1.0);
  nh.param("exploration/w_dir", ViewNode::w_dir_, -1.0);

  ViewNode::astar_.reset(new Astar);
  ViewNode::astar_->init(nh, edt_environment_);
  ViewNode::map_ = sdf_map_;

  double resolution_ = sdf_map_->getResolution();
  Eigen::Vector3d origin, size;
  sdf_map_->getFullMap(origin, size);
  ViewNode::raycaster_.reset(new RayCaster);
  ViewNode::raycaster_->setParams(resolution_, origin);


  // Initialize TSP par file
  ofstream par_file(ep_->tsp_dir_ + "/single.par");
  par_file << "PROBLEM_FILE = " << ep_->tsp_dir_ << "/single.tsp\n";
  par_file << "GAIN23 = NO\n";
  par_file << "OUTPUT_TOUR_FILE =" << ep_->tsp_dir_ << "/single.txt\n";
  par_file << "RUNS = 1\n";

  // Analysis
  // ofstream fout;
  // fout.open("/home/boboyu/Desktop/RAL_Time/frontier.txt");
  // fout.close();
}

// first: find current frontiers
// seconde: find a global tour and refine a local tour if needed
// third: find an astar path to the next goal point along the tour, and then shorten it
// forth: replanning a trajectory according to the path: polynomial to bspline and then optimize it
// forth: or maybe just do a kinodynamicReplanFromTo to the next goal point along the tour rather than astar
// fifth: plan for yaw tracking
int FastExplorationManager::planExploreMotion(
    const Vector3d& pos, const Vector3d& vel, const Vector3d& acc, const Vector3d& yaw) {
  ros::Time t1 = ros::Time::now();
  auto t2 = t1;
  ed_->views_.clear();

  std::cout << "start pos: " << pos.transpose() << "\nvel: " << vel.transpose()
            << "\nacc: " << acc.transpose() << "\nyaw: " << yaw.transpose() << std::endl;

  // Step1: Update frontier
  // ===================================================================
  std::cout<<"=========== Frontier ==========="<<std::endl;
  t1 = ros::Time::now();
  if (updateFrontier(pos) == NO_FRONTIER)
    return NO_FRONTIER;
  double frontier_searching_time = 1000.0*(ros::Time::now() - t1).toSec();
  ROS_WARN(
      "[exploreManager]:total frontier to visit: %d, viewpoint: %d, searching time(ms): %lf", 
      ed_->frontiers_cells_.size(), ed_->points_.size(), frontier_searching_time);
  
  // Step2: Explore a tour
  // ===================================================================
  std::cout<<"=========== Exploration ==========="<<std::endl;
  t1 = ros::Time::now();
  exploreTour(pos, vel, acc, yaw);
  if((next_viewpoint_to_visit-pos).norm()<0.1 && ed_->points_.size() == 1) return FAIL;
  double local_time = 1000.0*(ros::Time::now() - t1).toSec();
  ROS_WARN("[exploreManager]:Exploration tour searching & refining time(ms): %lf", local_time);

  // Step3: Plan bspline trajectory to the next viewpoint with astar initialization
  // ===================================================================
  std::cout<<"=========== Planning ==========="<<std::endl;
  if (planTrajectory(pos, vel, acc, yaw) == FAIL)
    return FAIL;

  double total = 1000.0*(ros::Time::now() - t2).toSec();
  ROS_WARN("[exploreManager]:Total time(ms): %lf", total);
  ROS_ERROR_COND(total > 0.1, "[exploreManager]:Total time too long!!!");

  return SUCCEED;
}

// Step 1: update frontiers
// =========================================
int FastExplorationManager::updateFrontier(const Vector3d& pos){
  // Search new frontier cells and allocate it as new frontiers
  frontier_finder_->updateFrontiers();
  // Find viewpoints (x,y,z,yaw) for each frontier on its cells
  frontier_finder_->searchViewpointsOfNewFrontiers();

  // get visible ones' info
  if (!ed_->frontiers_cells_.empty()) ed_->frontiers_cells_.clear();
  if (!ed_->dead_frontiers_.empty()) ed_->dead_frontiers_.clear();
  frontier_finder_->getCellsOfEachFrontier(ed_->frontiers_cells_, ed_->dead_frontiers_);
#ifdef DEBUG
  std::cout<<"[DEBUG]:get frontiers"<<std::endl;
#endif
//   if (!ed_->frontier_boxes_.empty()) ed_->frontier_boxes_.clear();
//   frontier_finder_->getBoundingBoxOfEachFrontier(ed_->frontier_boxes_);
// #ifdef DEBUG
//   std::cout<<"[DEBUG]:get frontier box"<<std::endl;
// #endif
  if (ed_->frontiers_cells_.empty()) {
    ROS_WARN("No coverable frontier.");
    return NO_FRONTIER;
  }

  // Now we got the new points_, yaws_, average_, frontiers_ 
  // Publish them
  if (!ed_->points_.empty()) ed_->points_.clear();
  if (!ed_->yaws_.empty()) ed_->yaws_.clear();
  if (!ed_->averages_.empty()) ed_->averages_.clear();
  frontier_finder_->getTopViewpointOfEachFrontier(ed_->points_, ed_->yaws_, ed_->averages_);
  // frontier_finder_->getFarEnoughTopViewpointOfEachFrontierFrom(pos, ed_->points_, ed_->yaws_, ed_->averages_, , ep_->min_candidate_dist_);

  for (int i = 0; i < ed_->points_.size(); ++i)
    ed_->views_.push_back(
        ed_->points_[i] + 2.0 * Vector3d(cos(ed_->yaws_[i]), sin(ed_->yaws_[i]), 0));// points_[*] is the Vector3d
  return SUCCEED;
  // publish points_, yaws_, average_, frontiers_ 
}

// Step 2: explore a global tour to visit all top viewpoints of frontier
// =========================================
void FastExplorationManager::exploreTour(
    const Vector3d& pos, const Vector3d& vel, const Vector3d& acc, const Vector3d& yaw){
  // search global and local tour and retrieve the next viewpoint
  if (ed_->points_.size() > 1) {
    // Find the global tour passing through all viewpoints
    // Create TSP and solve by LKH
    // Optimal tour is returned as indices of frontier
    vector<int> indices;
    findGlobalTour(pos, vel, yaw, indices);
#ifdef DEBUG
    std::cout<<"[DEBUG]:find a global tour which has pos num: "<<indices.size()<<std::endl;
#endif
    if (ep_->refine_local_) {
      // Do refinement for the next few viewpoints in the global tour
      // Idx of the first K frontier in optimal tour
      ed_->refined_ids_.clear();
      ed_->unrefined_points_.clear();
      int knum = min(int(indices.size()), ep_->refined_num_);
      for (int i = 0; i < knum; ++i) {
        auto tmp = ed_->points_[indices[i]];
        ed_->unrefined_points_.push_back(tmp);
        ed_->refined_ids_.push_back(indices[i]);
        if ((tmp - pos).norm() > ep_->refined_radius_ && ed_->refined_ids_.size() >= 2) break;
      }
#ifdef DEBUG
      std::cout<<"[DEBUG]:local refine candidate pos num: "<<ed_->refined_ids_.size()<<std::endl;
#endif
      // Get several(>=top_view_num_ & >max_decay_*) top viewpoints(n_points_, n_yaws) 
      // that are far enough(min_candidate_dist_) from the next K frontiers(refined_ids_)
      // (satisfy refined_radius_ and refined_num_) from current pos
      ed_->n_points_.clear();
      vector<vector<double>> n_yaws;
      vector<Eigen::Vector3d> n_averages;
      frontier_finder_->getTopViewpointsOf_From(
          pos, ed_->refined_ids_, ep_->top_view_num_, ep_->max_decay_, ed_->n_points_, n_yaws, n_averages, ep_->min_candidate_dist_);
      // frontier_finder_->getConditionalTopViewpointsFrom_OF(
      //     pos, ed_->refined_ids_, ep_->top_view_num_, ep_->max_decay_, ed_->n_points_, n_yaws, ep_->min_candidate_dist_);
#ifdef DEBUG
      std::cout<<"[DEBUG]:get local refine best candidate pos"<<std::endl;
#endif
      ed_->refined_points_.clear();
      ed_->refined_views_.clear();
      vector<double> refined_yaws;
      vector<Eigen::Vector3d> refined_averages;
      refineLocalTour(pos, vel, yaw, ed_->n_points_, n_yaws, n_averages, ed_->refined_points_, refined_yaws, refined_averages);
#ifdef DEBUG
      std::cout<<"[DEBUG]:get local refine sequence num: "<<ed_->refined_points_.size()<<std::endl;
#endif
      next_viewpoint_to_visit = ed_->refined_points_[0];
      next_yaw = refined_yaws[0];
      next_frontier_average = refined_averages[0];

      // Get marker for view visualization
      ed_->refined_views1_.clear();
      ed_->refined_views2_.clear();
      for (int i = 0; i < ed_->refined_points_.size(); ++i) {
        Vector3d view =
            ed_->refined_points_[i] + 2.0 * Vector3d(cos(refined_yaws[i]), sin(refined_yaws[i]), 0);
        ed_->refined_views_.push_back(view);
      }
      for (int i = 0; i < ed_->refined_points_.size(); ++i) {
        vector<Vector3d> v1, v2;
        frontier_finder_->percep_utils_->setPose(ed_->refined_points_[i], refined_yaws[i]);
        frontier_finder_->percep_utils_->getFOV(v1, v2);
        ed_->refined_views1_.insert(ed_->refined_views1_.end(), v1.begin(), v1.end());
        ed_->refined_views2_.insert(ed_->refined_views2_.end(), v2.begin(), v2.end());
      }
    } else {
      // Choose the next viewpoint from global tour
      next_viewpoint_to_visit = ed_->points_[indices[0]];
      next_yaw = ed_->yaws_[indices[0]];
      next_frontier_average = ed_->averages_[indices[0]];
    }
  } else if (ed_->points_.size() == 1) {
    // Only 1 destination, no need to find global tour through TSP
    std::cout << "[exploreTour]:only one to visit"<<std::endl;
    ed_->global_tour_.clear();
    ed_->global_tour_ = { pos, ed_->points_[0] };

    if (ep_->refine_local_) {
      // Find the min cost viewpoint for next frontier
      ed_->refined_ids_ = { 0 };
      ed_->unrefined_points_ = { ed_->points_[0] };
      ed_->n_points_.clear();
      vector<vector<double>> n_yaws;
      vector<Eigen::Vector3d> n_averages;
      frontier_finder_->getTopViewpointsOf_From(
          pos, ed_->refined_ids_, ep_->top_view_num_, ep_->max_decay_, ed_->n_points_, n_yaws, n_averages, ep_->min_candidate_dist_);
      // frontier_finder_->getConditionalTopViewpointsFrom_OF(
      //     pos, { 0 }, ep_->top_view_num_, ep_->max_decay_, ed_->n_points_, n_yaws, ep_->min_candidate_dist_);

      double min_cost = 100000;
      int min_cost_id = -1;
      vector<Vector3d> tmp_path;
      for (int i = 0; i < ed_->n_points_[0].size(); ++i) {
        auto tmp_cost = ViewNode::computeCost(
            pos, ed_->n_points_[0][i], yaw[0], n_yaws[0][i], vel, yaw[1], tmp_path);
        if (tmp_cost < min_cost) {
          min_cost = tmp_cost;
          min_cost_id = i;
        }
      }
      next_viewpoint_to_visit = ed_->n_points_[0][min_cost_id];
      next_yaw = n_yaws[0][min_cost_id];
      next_frontier_average = n_averages[0];
      ed_->refined_points_ = { next_viewpoint_to_visit };
      ed_->refined_tour_.clear();

      // Get marker for view visualization
      ed_->refined_views1_.clear();
      ed_->refined_views2_.clear();
      ed_->refined_views_ = { next_viewpoint_to_visit + 2.0 * Vector3d(cos(next_yaw), sin(next_yaw), 0) };
    } else {
      next_viewpoint_to_visit = ed_->points_[0];
      next_yaw = ed_->yaws_[0];
      next_frontier_average = ed_->averages_[0];
    }
  } else{
    ROS_ERROR("Empty destination.");
    // next_viewpoint_to_visit = ed_->points_[0];
    // next_yaw = ed_->yaws_[0];
    // next_frontier_average = ed_->averages_[0];
  }

  std::cout << "[exploreTour]:Next to visit: " << next_viewpoint_to_visit.transpose() << ", yaw: " << next_yaw << std::endl;

  ed_->next_viewpoint_to_visit = 
      Vector4d(next_viewpoint_to_visit[0], next_viewpoint_to_visit[1], next_viewpoint_to_visit[2], next_yaw);
 
  ed_->next_frontier_average = 
      Vector4d(next_frontier_average[0], next_frontier_average[1], next_frontier_average[2], next_yaw);

  // publish next_viewpoint, next_yaw
}

// if frontier candidates are not only one, find a global tour firstly
// solve a LKH TSP problem via solveTSPLKH
void FastExplorationManager::findGlobalTour(
    const Vector3d& cur_pos, const Vector3d& cur_vel, const Vector3d cur_yaw,
    vector<int>& indices) {
  auto t1 = ros::Time::now();

  // Get cost matrix for current state and clusters
  Eigen::MatrixXd cost_mat;
  frontier_finder_->updatePathAndCostAmongTopFrontierViewpoints();
  frontier_finder_->computeFullCostMatrixAmongTopFrontierViewpointsFrom(cur_pos, cur_vel, cur_yaw, cost_mat);
  const int dimension = cost_mat.rows();

  double mat_time = 1000.0*(ros::Time::now() - t1).toSec();
  t1 = ros::Time::now();
#ifdef DEBUG
    std::cout << "tsp start " << std::endl;
#endif
  // Write params and cost matrix to problem file
  ofstream prob_file(ep_->tsp_dir_ + "/single.tsp");
  // Problem specification part, follow the format of TSPLIB

  string prob_spec = "NAME : single\nTYPE : ATSP\nDIMENSION : " + to_string(dimension) +
      "\nEDGE_WEIGHT_TYPE : "
      "EXPLICIT\nEDGE_WEIGHT_FORMAT : FULL_MATRIX\nEDGE_WEIGHT_SECTION\n";

  // string prob_spec = "NAME : single\nTYPE : TSP\nDIMENSION : " + to_string(dimension) +
  //     "\nEDGE_WEIGHT_TYPE : "
  //     "EXPLICIT\nEDGE_WEIGHT_FORMAT : LOWER_ROW\nEDGE_WEIGHT_SECTION\n";

  prob_file << prob_spec;
  // prob_file << "TYPE : TSP\n";
  // prob_file << "EDGE_WEIGHT_FORMAT : LOWER_ROW\n";
  // Problem data part
  const int scale = 100;
  if (false) {
    // Use symmetric TSP
    for (int i = 1; i < dimension; ++i) {
      for (int j = 0; j < i; ++j) {
        int int_cost = cost_mat(i, j) * scale;
        prob_file << int_cost << " ";
      }
      prob_file << "\n";
    }

  } else {
    // Use Asymmetric TSP
    for (int i = 0; i < dimension; ++i) {
      for (int j = 0; j < dimension; ++j) {
        int int_cost = cost_mat(i, j) * scale;
        prob_file << int_cost << " ";
      }
      prob_file << "\n";
    }
  }

  prob_file << "EOF";
  prob_file.close();

  // Call LKH TSP solver
  solveTSPLKH((ep_->tsp_dir_ + "/single.par").c_str());
#ifdef DEBUG
    std::cout << "tsp finished " << std::endl;
#endif
  // Read optimal tour from the tour section of result file
  ifstream res_file(ep_->tsp_dir_ + "/single.txt");
  string res;
  while (getline(res_file, res)) {
    // Go to tour section
    if (res.compare("TOUR_SECTION") == 0) break;
  }

  if (false) {
    // Read path for Symmetric TSP formulation
    getline(res_file, res);  // Skip current pose
    getline(res_file, res);
    int id = stoi(res);
    bool rev = (id == dimension);  // The next node is virutal depot?

    while (id != -1) {
      indices.push_back(id - 2);
      getline(res_file, res);
      id = stoi(res);
    }
    if (rev) reverse(indices.begin(), indices.end());
    indices.pop_back();  // Remove the depot

  } else {
    // Read path for ATSP formulation
    while (getline(res_file, res)) {
      // Read indices of frontiers in optimal tour
      int id = stoi(res);
      if (id == 1)  // Ignore the current state
        continue;
      if (id == -1) break;
      indices.push_back(id - 2);  // Idx of solver-2 == Idx of frontier
    }
  }

  res_file.close();

  // Get the path of optimal tour from path matrix
  // ed_->global_tour_.clear();
  // frontier_finder_->getPathAllFrontiersFrom_Along(cur_pos, indices, ed_->global_tour_);

  double tsp_time = 1000.0*(ros::Time::now() - t1).toSec();
  ROS_WARN("[globalTour]:Mat time(ms): %lf, TSP time(ms): %lf", mat_time, tsp_time);
}

// if need, after global tour, refine a local tour according to the N top viewpoints from the next K frontiers
// DijkstraSearch: cur_pos -> N points -> N points ...
void FastExplorationManager::refineLocalTour(
    const Vector3d& cur_pos, const Vector3d& cur_vel, const Vector3d& cur_yaw,
    const vector<vector<Vector3d>>& n_points, const vector<vector<double>>& n_yaws, const vector<Eigen::Vector3d> n_averages,
    vector<Vector3d>& refined_pts, vector<double>& refined_yaws, vector<Eigen::Vector3d>& refined_averages) {
  double create_time, search_time, parse_time;
  auto t1 = ros::Time::now();

  // Create graph for viewpoints selection
  GraphSearch<ViewNode> g_search;
  vector<ViewNode::Ptr> last_group, cur_group;

  // Add the current state
  ViewNode::Ptr first(new ViewNode(cur_pos, cur_yaw[0],Eigen::Vector3d::Zero()));
  first->vel_ = cur_vel;
  g_search.addNode(first);
  last_group.push_back(first);
  ViewNode::Ptr final_node;

  // Add viewpoints
  std::cout <<"[refineTour]:Local tour graph: \neach frontier has viewpoints:";
  for (int i = 0; i < n_points.size(); ++i) { // for each frontier
    // Create nodes for viewpoints of one frontier
    for (int j = 0; j < n_points[i].size(); ++j) { // for each top viewpoint candidate
      ViewNode::Ptr node(new ViewNode(n_points[i][j], n_yaws[i][j], n_averages[i]));
      g_search.addNode(node);
      // Connect a node to nodes in last group
      for (auto nd : last_group)
        g_search.addEdge(nd->id_, node->id_);
      cur_group.push_back(node);

      // Only keep the first viewpoint of the last local frontier
      if (i == n_points.size() - 1) {
        final_node = node;
        break;
      }
    }
    // Store nodes for this group for connecting edges
    std::cout << cur_group.size() << ", ";
    last_group = cur_group;
    cur_group.clear();
  }
  std::cout << "" << std::endl;
  create_time = (ros::Time::now() - t1).toSec();
  t1 = ros::Time::now();

  // Search optimal sequence
  vector<ViewNode::Ptr> path;
  g_search.DijkstraSearch(first->id_, final_node->id_, path);
#ifdef DEBUG
  std::cout<<"[DEBUG]:Dijsktra local refine finish, get path num: "<<path.size()<<std::endl;
  for(auto elem:path)
    std::cout<<"("<<elem->pos_.transpose()<<"), ";
  std::cout<<std::endl;
#endif
  search_time = (ros::Time::now() - t1).toSec();
  t1 = ros::Time::now();

  // Return searched sequence
  for (int i = 1; i < path.size(); ++i) {
    Eigen::Vector3d dir3d = path[i]->pos_ - cur_pos;
    // if (dir3d.norm() < ep_->min_candidate_dist_) continue; // skip too close
    double dir_diff = fabs(atan2(dir3d(1), dir3d(0))-cur_yaw[0]);
    dir_diff = dir_diff > M_PI ? 2*M_PI-dir_diff : dir_diff;
    // std::cout<<"cur_yaw: "<<cur_yaw[0]<<
    //           " , target yaw: "<<atan2(dir3d(1), dir3d(0))<<
    //           ", diff: "<<dir_diff<<std::endl;
    if(dir_diff>M_PI/2) continue;
    refined_pts.push_back(path[i]->pos_);
    refined_yaws.push_back(path[i]->yaw_);
    refined_averages.push_back(path[i]->average_);
  }
  if(refined_pts.empty()){
    ROS_WARN("======= all viewpoint to visit too close =========");
    refined_pts.push_back(path[1]->pos_);
    refined_yaws.push_back(path[1]->yaw_);
    refined_averages.push_back(path[1]->average_);
  }

  // Extract optimal local tour (for visualization)
  ed_->refined_tour_.clear();
  ed_->refined_tour_.push_back(cur_pos);
  ViewNode::astar_->lambda_heu_ = 1.0;
  ViewNode::astar_->setResolution(0.2);
  for (auto pt : refined_pts) {
    vector<Vector3d> path;
    if (ViewNode::searchPath(ed_->refined_tour_.back(), pt, path))
      ed_->refined_tour_.insert(ed_->refined_tour_.end(), path.begin(), path.end());
    else
      ed_->refined_tour_.push_back(pt);
  }
  ViewNode::astar_->lambda_heu_ = 10000;

  parse_time = (ros::Time::now() - t1).toSec();
  // ROS_WARN("create: %lf, search: %lf, parse: %lf", create_time, search_time, parse_time);
}

// Step 3: plan a trajectory to arrive the next viewpoint on the tour
// =========================================
int FastExplorationManager::planTrajectory(
    const Vector3d& pos, const Vector3d& vel, const Vector3d& acc, const Vector3d& yaw){
  ros::Time tic = ros::Time::now();
  static bool first_plan = true;
  // Compute time lower bound of yaw and use in trajectory generation
  double diff = fabs(next_yaw - yaw[0]);
  double time_lb = min(diff, 2 * M_PI - diff) / ViewNode::yd_;

  // Plan a astar path to the next viewpoint to visit
  planner_manager_->path_finder_->reset();
  if (planner_manager_->path_finder_->search(pos, next_viewpoint_to_visit) != Astar::REACH_END) {
    ROS_ERROR("No path to next viewpoint");
    return FAIL;
  }
  ed_->path_to_goal_ = planner_manager_->path_finder_->getPath();
  shortenPath(ed_->path_to_goal_);

  // Replan a bspline trajectory with a initial path
  const double too_close_along_path = 1.5;
  const double too_far_along_path = 5.0;
  const double truncated_length_along_path = 5.0;
  const double len = Astar::pathLength(ed_->path_to_goal_);
  if (len < too_close_along_path) {
    std::cout<<"[pathSearching]:Near goal."<<std::endl;
    // Next viewpoint is very close, no need to search kinodynamic path, just use waypoints-based
    // optimization
    planner_manager_->waypointsReplanAlongTour(ed_->path_to_goal_, vel, acc, time_lb);
    ed_->next_goal_ = next_viewpoint_to_visit;

  } else if (first_plan || len > too_far_along_path) {
    // first_plan = false; // kinodynamicReplan has some unknown error
    // Next viewpoint is far away, select intermediate goal on geometric path (this also deal with
    // dead end)
    std::cout << "[pathSearching]:Far goal." << std::endl;
    double len2 = 0.0;
    vector<Eigen::Vector3d> truncated_path = { ed_->path_to_goal_.front() };
    for (int i = 1; i < ed_->path_to_goal_.size() && len2 < truncated_length_along_path; ++i) {
      auto cur_pt = ed_->path_to_goal_[i];
      double step = (cur_pt - truncated_path.back()).norm();
      if (step > 1e-3){
        len2 += step;
        truncated_path.push_back(cur_pt);
      }
    }
    // Ensure at least three points in the path
    if (truncated_path.size() == 2)
      truncated_path.insert(truncated_path.begin() + 1, 0.5 * (truncated_path[0] + truncated_path[1]));
    ed_->next_goal_ = truncated_path.back();

    planner_manager_->waypointsReplanAlongTour(truncated_path, vel, acc, time_lb);
  } else {
    // Search kino path to exactly next viewpoint and optimize
    std::cout << "[pathSearching]:Mid goal" << std::endl;
    ed_->next_goal_ = next_viewpoint_to_visit;

    if (!planner_manager_->kinodynamicReplanFromTo(
            pos, vel, acc, next_viewpoint_to_visit, Vector3d(0, 0, 0), time_lb))
      return FAIL;
  }

  if (planner_manager_->local_data_.position_traj_.getTimeSum() < time_lb - 0.1)
    ROS_ERROR("Lower bound not satified!");

  double traj_plan_time = (ros::Time::now() - tic).toSec();
  tic = ros::Time::now();

  // Plan a yaw tracking trajectory
  // planner_manager_->planYawFromToWith(yaw, next_yaw); // yaw smoothing to nex_yaw
  // planner_manager_->planYawFromToWith(yaw, next_yaw, true, ep_->relax_time_); // yaw tracking traj and then smoothing end to next yaw
  planner_manager_->planYawToward(yaw, next_frontier_average, ep_->relax_time_); // yaw tracking towards target

  double yaw_time = (ros::Time::now() - tic).toSec();
  ROS_WARN("[trajPlan]:Trajectory planning time: %lf, yaw time: %lf", traj_plan_time, yaw_time);
  return SUCCEED;
}

// shorten path aster found, in order to replan it as bspline
void FastExplorationManager::shortenPath(vector<Vector3d>& path) {
  if (path.empty()) {
    ROS_ERROR("Empty path to shorten");
    return;
  }
  // Shorten the tour, only critical intermediate points are reserved.
  const double dist_thresh = 2.0; // should equal to truncated_length_along_path or not ????
  vector<Vector3d> short_tour = { path.front() };
  for (int i = 1; i < path.size() - 1; ++i) {
    if ((path[i] - short_tour.back()).norm() > dist_thresh)
      short_tour.push_back(path[i]);
    else {
      // Add path[i] if collision occur when direct to next
      ViewNode::raycaster_->setInput(short_tour.back(), path[i + 1]);
      Eigen::Vector3i idx;
      while (ViewNode::raycaster_->nextId(idx) && ros::ok()) {
        if (edt_environment_->sdf_map_->getInflateOccupancy(idx) == 1 ||
            edt_environment_->sdf_map_->getOccupancy(idx) == SDFMap::UNKNOWN) {
          short_tour.push_back(path[i]);
          break;
        }
      }
    }
  }
  if ((path.back() - short_tour.back()).norm() > 1e-3) short_tour.push_back(path.back());

  // Ensure at least three points in the path
  if (short_tour.size() == 1){
    ROS_ERROR("[shortenPath]: only 1 point, copy twice!!!\n");
    std::cout<<"path size: "<<path.size()<<": \n"<<path.front().transpose()<<"\n"<<path.back().transpose()<<std::endl;
    short_tour.push_back(short_tour.back());
    short_tour.push_back(short_tour.back());
  }else if (short_tour.size() == 2){
    std::cout<<"[shortenPath]: only 2 points, interpolation 1!!! path: "<<path.size()<<": \n"<<path.front().transpose()<<"\n"<<path.back().transpose()<<std::endl;
    short_tour.insert(short_tour.begin() + 1, 0.5 * (short_tour[0] + short_tour[1]));
  }
  path = short_tour;
}

void FastExplorationManager::getPath(vector<Vector3d>& path){
  path = ed_->path_to_goal_;
}

void FastExplorationManager::getGoal(Eigen::Vector3d& goal){
  goal = ed_->next_goal_;
}

void FastExplorationManager::getViewpoint(Eigen::Vector4d& next_viewpoint){
  next_viewpoint = ed_->next_viewpoint_to_visit;
}

void FastExplorationManager::getFrontierInfo(Eigen::Vector4d& next_frontier_info){
  next_frontier_info = ed_->next_frontier_average;
}

void FastExplorationManager::getFrontier(vector<vector<Vector3d>>& frontier_, vector<vector<Vector3d>>& dead_frontiers_){
  frontier_ = ed_->frontiers_cells_;
  dead_frontiers_ = ed_->dead_frontiers_;
}

}  // namespace fast_planner
