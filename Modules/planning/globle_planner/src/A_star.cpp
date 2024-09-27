#include "A_star.h"

using namespace std;
using namespace Eigen;

namespace Global_Planning
{

void Astar::init(ros::NodeHandle& nh)
{
  nh.param("global_planner/is_2D", is_2D, 0);
  nh.param("global_planner/2D_fly_height", fly_height, 1.5);
  nh.param("astar/lambda_heu", lambda_heu_, 2.0); 
  nh.param("astar/allocate_num", max_search_num, 100000);
  nh.param("map/resolution", resolution_, 0.2);
  nh.param("global_planner/safe_distance", safe_distance, 0.05); 

  tie_breaker_ = 1.0 + 1.0 / max_search_num;

  this->inv_resolution_ = 1.0 / resolution_;

  has_global_point = false;
  path_node_pool_.resize(max_search_num);
  for (int i = 0; i < max_search_num; i++){
    path_node_pool_[i] = new PathNode;
  }

  use_node_num_ = 0;
  iter_num_ = 0;

  Occupy_map_ptr.reset(new Occupy_map);
  Occupy_map_ptr->init(nh);

  origin_ =  Occupy_map_ptr->min_range_;
  map_size_3d_ = Occupy_map_ptr->max_range_ - Occupy_map_ptr->min_range_;
}

int Astar::search(Eigen::Vector3d start_pt, Eigen::Vector3d start_v, Eigen::Vector3d start_a,
                             Eigen::Vector3d end_pt, Eigen::Vector3d end_v, bool init, bool dynamic, double time_start)
{
  if(Occupy_map_ptr->getOccupancy(end_pt))
  {
    ROS_ERROR("Goal point is occupied.");
    return NO_PATH;
  }

  ros::Time tic = ros::Time::now();
  
  goal_pos = end_pt;
  Eigen::Vector3i end_index = posToIndex(end_pt);

  PathNodePtr cur_node = path_node_pool_[0];
  cur_node->parent = NULL;
  cur_node->position = start_pt;
  cur_node->index = posToIndex(start_pt);
  cur_node->g_score = 0.0;
  cur_node->f_score = lambda_heu_ * getDiagHeu(cur_node->position, end_pt);
  cur_node->node_state = IN_OPEN_SET;
   
  open_set_.push(cur_node);
  expanded_nodes_.insert(cur_node->index, cur_node);
  use_node_num_ += 1;

  PathNodePtr terminate_node = NULL;

  while (!open_set_.empty())
  {
    cur_node = open_set_.top();

    bool reach_end = abs(cur_node->index(0) - end_index(0)) <= 1 && 
                     abs(cur_node->index(1) - end_index(1)) <= 1 &&
                     abs(cur_node->index(2) - end_index(2)) <= 1;
    if (reach_end){
      terminate_node = cur_node;
      retrievePath(terminate_node);
      printf("Astar take time %f s. \n", (ros::Time::now()-tic).toSec());
      return REACH_END;
    }

    /* ---------- pop node and add to close set ---------- */
    open_set_.pop();
    cur_node->node_state = IN_CLOSE_SET;  // in expand set
    iter_num_ += 1;

    /* ---------- init neighbor expansion ---------- */
    Eigen::Vector3d cur_pos = cur_node->position;
    Eigen::Vector3d expand_node_pos;

    vector<Eigen::Vector3d> inputs;

    /* ---------- expansion loop ---------- */
    Eigen::Vector3d d_pos;
    for (double dx = -resolution_; dx <= resolution_ + 1e-3; dx += resolution_)
      for (double dy = -resolution_; dy <= resolution_ + 1e-3; dy += resolution_)
        for (double dz = -resolution_; dz <= resolution_ + 1e-3; dz += resolution_){
          if(is_2D)
            d_pos << dx,dy,0.0;
          else
            d_pos << dx, dy, dz;

          if (d_pos.norm() < 1e-3)
            continue;

          expand_node_pos = cur_pos + d_pos;
          /* ---------- check if nbr_pos in feasible space ---------- */
          if (//Occupy_map_ptr->getOccupancy(expand_node_pos) ||
              !Occupy_map_ptr->check_safety(expand_node_pos, safe_distance) ||
              !Occupy_map_ptr->isInMap(expand_node_pos))
            continue;

          /* ---------- check if nbr_pos not in close set ---------- */
          Eigen::Vector3i d_pos_id;
          d_pos_id << int(dx/resolution_), int(dy/resolution_), int(dz/resolution_);
          Eigen::Vector3i expand_node_id = d_pos_id + cur_node->index;
          PathNodePtr expand_node = expanded_nodes_.find(expand_node_id);
          if (expand_node != NULL && expand_node->node_state == IN_CLOSE_SET)
            continue;

          /* ---------- add neighbor if available ---------- */
          double tmp_g_score, tmp_f_score;
          tmp_g_score = d_pos.squaredNorm() + cur_node->g_score;
          tmp_f_score = tmp_g_score + lambda_heu_ * getDiagHeu(expand_node_pos, end_pt);

          if (expand_node == NULL){
            // first visit, add into list
            expand_node = path_node_pool_[use_node_num_++];
            if (use_node_num_ == max_search_num){
              ROS_WARN("reach the max_search_num.");
              return NO_PATH;
            }
            expand_node->index = expand_node_id;
            expand_node->position = expand_node_pos;
            expand_node->f_score = tmp_f_score;
            expand_node->g_score = tmp_g_score;
            expand_node->parent = cur_node;
            expand_node->node_state = IN_OPEN_SET;

            open_set_.push(expand_node);
            expanded_nodes_.insert(expand_node_id, expand_node);
          }else if (expand_node->node_state == IN_OPEN_SET){
            if (tmp_g_score < expand_node->g_score){
              // not first visit, but current visit batter then the previous visit, replace it
              // expand_node->index = expand_node_id;
              expand_node->position = expand_node_pos;
              expand_node->f_score = tmp_f_score;
              expand_node->g_score = tmp_g_score;
              expand_node->parent = cur_node;
            }
          }

          if (is_2D)
            break;
        }
  }
  ROS_WARN("open set empty. no path");
  return NO_PATH;
}
}
