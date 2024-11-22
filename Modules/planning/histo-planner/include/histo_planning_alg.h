#ifndef HISTO_PLANNING_ALG
#define HISTO_PLANNING_ALG

#include <Eigen/Eigen>
#include <iostream>

#include <ros/ros.h>
#include <visualization_msgs/Marker.h>
#include <geometry_msgs/PoseStamped.h>
#include <std_msgs/Empty.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

using namespace std;

namespace Histo_Planning{

class histo_planning_alg{
public:
    histo_planning_alg(){}
    ~histo_planning_alg(){}
    virtual void init(ros::NodeHandle& nh) = 0;
    virtual void set_odom(nav_msgs::Odometry cur_odom) = 0;
    virtual void set_local_map_pcl(pcl::PointCloud<pcl::PointXYZ>::Ptr &pcl_ptr) = 0;
    
    virtual int generate(Eigen::Vector3d &start_pos, Eigen::Vector3d &start_vel, Eigen::Vector3d &goal, Eigen::Vector3d &desired) = 0;
    virtual vector<Eigen::Vector3d> getSamples(Eigen::Vector3d &start_pos, Eigen::Vector3d &start_vel, Eigen::Vector3d &start_acc, Eigen::Vector3d &goal, Eigen::Vector3d &goal_vel, Eigen::Vector3d &goal_acc, Eigen::Vector3d &guide_point, double& ts) = 0;
    
    virtual double getDistWithGrad(Eigen::Vector3d pos, Eigen::Vector3d &grad) = 0;
    virtual double getDist(Eigen::Vector3d pos, Eigen::Vector3d &closest) = 0;
    
    virtual double** get_Histogram_3d() const = 0;
    virtual double** get_Obstacle_3d() const = 0;
    virtual int get_Hcnt() const = 0;
    virtual int get_Vcnt() const = 0;
    virtual double get_Hres() const = 0;
    virtual double get_Vres() const = 0;

    typedef shared_ptr<histo_planning_alg> Ptr;
};

}

#endif 
