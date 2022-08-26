#ifndef HIST_H
#define HIST_H

#include <ros/ros.h>
#include <Eigen/Eigen>
#include <iostream>
#include <algorithm>
#include <iostream>

#include <geometry_msgs/PoseStamped.h>
#include <std_msgs/Empty.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include "tools.h"
#include "local_planning_alg.h"
#include "message_utils.h"

using namespace std;

namespace Local_Planning
{

extern ros::Publisher message_pub;

class HIST: public local_planning_alg
{
private:
    //　参数
    double sensor_max_range, forbidden_range, safe_distance, forbidden_plus_safe_distance;
    double ground_height, ceil_height;
	double max_distance;
    double limit_v_norm;
    Eigen::Vector3d best_dir;
    
    double  Hres;
    int Hcnt;  // 直方图横向个数
    double  Vres;
    int Vcnt;  // 直方图纵向个数

	// bool 参数
	bool gen_guide_point;
    bool has_local_map_, has_odom_, has_best_dir;
    bool is_2D, isCylindrical, isSpherical;


	// Histogram
    double* Histogram_2d;
    double** Histogram_3d;

    pcl::PointCloud<pcl::PointXYZ> latest_local_pcl_;
    sensor_msgs::PointCloud2ConstPtr  local_map_ptr_;
    nav_msgs::Odometry cur_odom_;

    void PolarCoordinateHist(double angle_cen, double angle_range, double val);
    void CylindricalCoordinateHist(double hor_angle_cen, double ver_angle_cen, double idx_range, double hor_obs_dist, double val);
    void SphericalCoordinateHist(double hor_angle_cen, double ver_angle_cen, double angle_range, double val);

public:

    void set_odom(nav_msgs::Odometry cur_odom);
    void set_local_map_pcl(pcl::PointCloud<pcl::PointXYZ>::Ptr &pcl_ptr);
    int generate(Eigen::Vector3d  &goal, Eigen::Vector3d &desired);
    void init(ros::NodeHandle& nh);

    HIST(){}
    ~HIST(){
        delete Histogram_2d;
        if (is_2D){
		    for(int i = 0; i < Vcnt; i++)
				delete[] Histogram_3d[i];
			delete[] Histogram_3d;
        }
    }

    typedef shared_ptr<HIST> Ptr;

};

}

#endif 
