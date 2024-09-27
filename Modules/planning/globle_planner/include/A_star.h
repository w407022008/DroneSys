#ifndef _ASTAR_H
#define _ASTAR_H

#include <ros/ros.h>
#include <Eigen/Eigen>
#include <iostream>
#include <queue>
#include <string>
#include <unordered_map>
#include <sstream>

#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Path.h>

#include "occupy_map.h"
#include "tools.h"
#include "global_planning_alg.h"

#define NODE_NAME "Global_Planner [Astar]"


using namespace std;

namespace Global_Planning
{

class Astar: public global_planning_alg
{
private:
    std::vector<PathNodePtr> path_node_pool_;
    int use_node_num_, iter_num_;
    NodeHashTable expanded_nodes_;
    std::priority_queue<PathNodePtr, std::vector<PathNodePtr>, NodeComparator> open_set_;
    std::vector<PathNodePtr> path_nodes_;  

    double lambda_heu_;
    int max_search_num;
    double tie_breaker_;
    int is_2D;
    double fly_height;

    Eigen::Vector3d goal_pos;

    std::vector<int> occupancy_buffer_;  
    double resolution_, inv_resolution_, safe_distance;
    Eigen::Vector3d origin_, map_size_3d_;
    bool has_global_point;

    Eigen::Vector3i posToIndex(Eigen::Vector3d pt){
        Eigen::Vector3i idx;
        idx << floor((pt(0) - origin_(0)) * inv_resolution_), 
                floor((pt(1) - origin_(1)) * inv_resolution_),
                floor((pt(2) - origin_(2)) * inv_resolution_);
        return idx;
    };

    Eigen::Vector3d indexToPos(Eigen::Vector3i id){
        Eigen::Vector3d pos;
        for (int i = 0; i < 3; ++i)
            pos(i) = (id(i) + 0.5) * resolution_ + origin_(i);
        return pos;
    };

    void indexToPos(Eigen::Vector3i id, Eigen::Vector3d &pos){
        for (int i = 0; i < 3; ++i)
            pos(i) = (id(i) + 0.5) * resolution_ + origin_(i);
    };

    void retrievePath(PathNodePtr end_node){
        PathNodePtr cur_node = end_node;
        path_nodes_.push_back(cur_node);
        while (cur_node->parent != NULL){
            cur_node = cur_node->parent;
            path_nodes_.push_back(cur_node);
        }
        reverse(path_nodes_.begin(), path_nodes_.end());
    };

    double getDiagHeu(Eigen::Vector3d x1, Eigen::Vector3d x2){
        double dx = fabs(x1(0) - x2(0));
        double dy = fabs(x1(1) - x2(1));
        double dz = fabs(x1(2) - x2(2));
        double h;
        double diag = min(min(dx, dy), dz);
        dx -= diag;
        dy -= diag;
        dz -= diag;

        if (dx < 1e-4) {
            h = 1.0 * sqrt(3.0) * diag + sqrt(2.0) * min(dy, dz) + 1.0 * abs(dy - dz);
        }
        if (dy < 1e-4) {
            h = 1.0 * sqrt(3.0) * diag + sqrt(2.0) * min(dx, dz) + 1.0 * abs(dx - dz);
        }
        if (dz < 1e-4) {
            h = 1.0 * sqrt(3.0) * diag + sqrt(2.0) * min(dx, dy) + 1.0 * abs(dx - dy);
        }
        return tie_breaker_ * h;
    };

    double getManhHeu(Eigen::Vector3d x1, Eigen::Vector3d x2){
        double dx = fabs(x1(0) - x2(0));
        double dy = fabs(x1(1) - x2(1));
        double dz = fabs(x1(2) - x2(2));
        return tie_breaker_ * (dx + dy + dz);
    };

    double getEuclHeu(Eigen::Vector3d x1, Eigen::Vector3d x2){
        return tie_breaker_ * (x2 - x1).norm();
    };

    // shorten path aster found, in order to replan it as bspline
    void shortenPath(vector<Eigen::Vector3d>& path) {
        if (path.empty()) {
            ROS_ERROR("Empty path to shorten");
            return;
        }
        // Shorten the tour, only critical intermediate points are reserved.
        const double dist_thresh = 5.0; // should equal to truncated_length_along_path or not ????
        vector<Eigen::Vector3d> short_path = { path.front() };
        for (int i = 1; i < path.size() - 1; ++i) {
                // Add path[i] if collision occur when direct to next
                Eigen::Vector3i curId = posToIndex(short_path.back());
                Eigen::Vector3i goalId = posToIndex(path[i + 1]);
                double distId = (curId-goalId).norm();
                Eigen::Vector3d step = (goalId-curId).cast<double>();
                step.normalize();
                Eigen::Vector3d curId3d = curId.cast<double>();
                Eigen::Vector3d lastId = curId3d;
                Eigen::Vector3d nextId = lastId + step;
                while(ros::ok()){
                    while((nextId - lastId).cwiseAbs().maxCoeff()<1.0){
                        nextId = nextId + step;
                    }
                    Eigen::Vector3i nextId3i = nextId.cast<int>();
                    if(//Occupy_map_ptr->getOccupancy(nextId3i) 
                        !Occupy_map_ptr->check_safety(nextId3i, safe_distance) 
                        || !Occupy_map_ptr->isInMap(nextId3i)){
                        short_path.push_back(path[i]);
                        break;
                    }
                    lastId = nextId;
                    nextId = nextId + step;

                    if((nextId - goalId.cast<double>()).maxCoeff()<1.0){
                        break;
                    }
                }
            
        }
        if ((path.back() - short_path.back()).norm() > 1e-3) short_path.push_back(path.back());

        path = short_path;
    }

    // interpolate path
    void interpolatePath(vector<Eigen::Vector3d>& path) {
        if (path.empty()) {
            ROS_ERROR("Empty path to shorten");
            return;
        }
        
        vector<Eigen::Vector3d> long_path = { path.front() };
        for (auto pos_ : path) {
            Eigen::Vector3d step = pos_ - long_path.back();
            double dist = step.norm();
            if (dist < 1.5*resolution_)
                long_path.push_back(pos_);
            else{
                step.normalize();
                for (int j=1; (j)*resolution_<dist;++j){
                    Eigen::Vector3d curPos = long_path.back()+resolution_*step;
                    long_path.push_back(curPos);
                }
                long_path.push_back(pos_);
            }
        }
        path = long_path;
        
    }

public:

    Astar(){}
    ~Astar(){
        for (int i = 0; i < max_search_num; i++)
        {
            // delete表示释放堆内存
            delete path_node_pool_[i];
        }
    };
    
    // std::vector<Eigen::Vector3d> path_pos;

    void init(ros::NodeHandle& nh);

    int search(Eigen::Vector3d start_pt, Eigen::Vector3d start_vel, Eigen::Vector3d start_acc,
            Eigen::Vector3d end_pt, Eigen::Vector3d end_vel, bool init = false, bool dynamic = false,
            double time_start = -1.0);

    void reset(){
        expanded_nodes_.clear();
        path_nodes_.clear();

        std::priority_queue<PathNodePtr, std::vector<PathNodePtr>, NodeComparator> empty_queue;
        open_set_.swap(empty_queue);

        for (int i = 0; i < use_node_num_; i++){
            PathNodePtr node = path_node_pool_[i];
            node->parent = NULL;
            node->node_state = NOT_EXPAND;
        }

        use_node_num_ = 0;
        iter_num_ = 0;
    }
    
    bool check_safety(Eigen::Vector3d &cur_pos, double safe_distance){
        return Occupy_map_ptr->check_safety(cur_pos, safe_distance);
    };
    
    std::vector<Eigen::Vector3d> getPath(){
        vector<Eigen::Vector3d> path;
        for (uint i = 0; i < path_nodes_.size(); ++i)
        {
            path.push_back(path_nodes_[i]->position);
        }
        path.push_back(goal_pos);
        shortenPath(path);
        interpolatePath(path);
        return path;
    }

    nav_msgs::Path get_ros_path(){
        path_pos = getPath();
        nav_msgs::Path path;

        path.header.frame_id = "world";
        path.header.stamp = ros::Time::now();
        path.poses.clear();

        geometry_msgs::PoseStamped path_i_pose;
        for (uint i=0; i<path_pos.size(); ++i){   
            path_i_pose .header.frame_id = "world";
            path_i_pose.pose.position.x = path_pos[i][0];
            path_i_pose.pose.position.y = path_pos[i][1];
            path_i_pose.pose.position.z = path_pos[i][2];
            path.poses.push_back(path_i_pose);
        }

        path_i_pose .header.frame_id = "world";
        path_i_pose.pose.position.x = goal_pos[0];
        path_i_pose.pose.position.y = goal_pos[1];
        path_i_pose.pose.position.z = goal_pos[2];
        path.poses.push_back(path_i_pose);

        return path;
    };

    std::vector<PathNodePtr> getVisitedNodes(){
        vector<PathNodePtr> visited;
        visited.assign(path_node_pool_.begin(), path_node_pool_.begin() + use_node_num_ - 1);
        return visited;
    }

    typedef shared_ptr<Astar> Ptr;
};
}
#endif
