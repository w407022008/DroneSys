#ifndef POINTS_FILTER_H
#define POINTS_FILTER_H

#include <ros/ros.h>
#include <Eigen/Eigen>
#include <iostream>
#include <algorithm>

#include <chrono>

#include <geometry_msgs/PoseStamped.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/LaserScan.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/CameraInfo.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/io.h>
#include <pcl/conversions.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl_ros/transforms.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf/transform_listener.h>
#include <tf/message_filter.h>

using namespace std;

namespace Points_Filter
{

extern ros::Publisher message_pub;

class PointsFilter
{
private:
  ros::NodeHandle points_filter_nh;

  /* common param */
  int map_input;
  string frame_name, object_link_name;
  bool flag_pcl_ground_removal, downsampling, spatial, concatenate, is_rgbd, is_lidar;
  float resolution, sensor_max_range, max_ground_height;
  pcl::PointCloud<pcl::PointXYZ> latest_local_pcl_, local_point_cloud; // point cloud
  
  /* camera parameters */
  double cx, cy, fx, fy;
  int depth_height, depth_width;
  cv::Mat depth_image_;
  
  /* depth image projection filtering */
  double dist_min;
  int cut_edge, interval;
  
  void Callback_2dlaserscan(const sensor_msgs::LaserScanConstPtr &msg);
  void Callback_3dpointcloud(const sensor_msgs::PointCloud2ConstPtr &msg);
  void Callback_depthimage(const sensor_msgs::ImageConstPtr &img);
  void Callback_depthinfo(const sensor_msgs::CameraInfoConstPtr &info);

  ros::Subscriber local_point_clound_sub, camera_info_sub;
  tf::TransformListener tfListener;
  ros::Publisher point_cloud_pub;

public:
  PointsFilter(void):
  points_filter_nh("~") {}~PointsFilter(){}
        
  void init(ros::NodeHandle& nh);
  
};

}

#endif 
