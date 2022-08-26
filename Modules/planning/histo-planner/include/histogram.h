#ifndef HIST_H
#define HIST_H

#include <ros/ros.h>
#include <Eigen/Eigen>
#include <iostream>
#include <chrono>
#include <thread>
#include <algorithm>
#include <iostream>
#include <mutex>

#include <geometry_msgs/PoseStamped.h>
#include <std_msgs/Empty.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>

#include <pcl/io/pcd_io.h>

#include "string.h"
#include "math.h"
#include "ctime"
#include "histo_planning_alg.h"
#include "message_utils.h"

using namespace std;

struct PointXYZT    // enforce SSE padding for correct memory alignment
{
  PCL_ADD_POINT4D;                  // preferred way of adding a XYZ+padding
  float t;
  PCL_MAKE_ALIGNED_OPERATOR_NEW     // make sure our new allocators are aligned
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZT,           // here we assume a XYZ + "t" (as fields)
                                   (float, x, x)
                                   (float, y, y)
                                   (float, z, z)
                                   (float, t, t)
)

namespace Histo_Planning
{

class HIST: public histo_planning_alg
{
private:
    
  ros::Publisher point_cloud_pub;
  double sensor_max_range, forbidden_range, safe_distance, forbidden_plus_safe_distance;
  double ground_height, ceil_height;
  double max_distance;
  double limit_v_norm;
  double min_value, _pow_;
  double _std_;
  double min_vel_default;
  int piecewise_interpolation_num;
  Eigen::Vector3d best_dir;
  
  double  Hres;
  int Hcnt; 
  double  Vres;
  int Vcnt;

  bool gen_guide_point;
  bool has_local_map_, has_odom_, has_best_dir;
  bool is_2D, isCylindrical, isSpherical;

  std::chrono::time_point<std::chrono::system_clock> start_pcl, start_gen;

  std::mutex mutex;
  
  // Histogram
  Eigen::Vector3d capture_pos;
  double** Histogram_3d;
  double** Obstacle_3d;
  double*** Obs_buff_3d;
  double*** Env_3d;
  double** Weights_3d;

  pcl::PointCloud<pcl::PointXYZ> latest_local_pcl_;
  pcl::PointCloud<PointXYZT> latest_pcl_xyzt_buff;
  // nav_msgs::Odometry cur_odom_;

  void PolarCoordinateHist(double angle_cen, double angle_range, double val);
  void CylindricalCoordinateHist(double hor_angle_cen, double ver_idx_cen, double hor_obs_dist);
  void SphericalCoordinateHist(double hor_angle_cen, double ver_angle_cen, double angle_range, double obs_dist);

public:
	
  void init(ros::NodeHandle& nh);

  void set_odom(nav_msgs::Odometry cur_odom);
  void set_local_map_pcl(pcl::PointCloud<pcl::PointXYZ>::Ptr &pcl_ptr);
    
  int generate(Eigen::Vector3d &start_pos, Eigen::Vector3d &start_vel, Eigen::Vector3d &goal, Eigen::Vector3d &desired);
    
  vector<Eigen::Vector3d> getSamples(Eigen::Vector3d &start_pos, Eigen::Vector3d &start_vel, Eigen::Vector3d &start_acc, Eigen::Vector3d &goal, Eigen::Vector3d &goal_vel, Eigen::Vector3d &goal_acc, Eigen::Vector3d &guide_point, double& ts);
    
  double getDistWithGrad(Eigen::Vector3d pos, Eigen::Vector3d &grad);
  double getDist(Eigen::Vector3d pos, Eigen::Vector3d &closest);
    
  double** get_Histogram_3d() const{return Histogram_3d;}
  double** get_Obstacle_3d() const{return Obstacle_3d;}

  int get_Vcnt() const{return Vcnt;}
  int get_Hcnt() const{return Hcnt;}
  double get_Hres() const{return Hres;}
  double get_Vres() const{return Vres;}

  HIST(){}
  ~HIST(){
    for(int i = 0; i < Vcnt; i++)
      delete[] Histogram_3d[i];
    delete[] Histogram_3d;
  }

  typedef shared_ptr<HIST> Ptr;

};

}

#endif 
