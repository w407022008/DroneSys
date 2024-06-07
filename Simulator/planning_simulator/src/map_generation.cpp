
#include <ros/ros.h>
#include <ros/console.h>
#include <math.h>
#include <iostream>
#include <Eigen/Eigen>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_conversions/pcl_conversions.h>

#include <sensor_msgs/PointCloud2.h>
#include "drone_msgs/DroneState.h"

using namespace std;

void MapGenerate();

bool _map_ok = false;
double sense_range;
double _resolution;

pcl::KdTreeFLANN<pcl::PointXYZ> kdtreeLocalMap;

pcl::PointCloud<pcl::PointXYZ> cloudMap;
sensor_msgs::PointCloud2 localENUMap_pcd;

ros::Subscriber drone_state_sub;
ros::Publisher local_ENU_map_pub;

void drone_state_cb(const drone_msgs::DroneState::ConstPtr &msg)
{
  if (!_map_ok) return;
  
  pcl::PointCloud<pcl::PointXYZ> localEUNMap;

  pcl::PointXYZ searchPoint(msg->position[0], msg->position[1], msg->position[2]);
  vector<int> pointIdxRadiusSearch;
  vector<float> pointRadiusSquaredDistance;

  pcl::PointXYZ pt;
  
  if (kdtreeLocalMap.radiusSearch(searchPoint, sense_range, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0)
  {
    for (size_t i = 0; i < pointIdxRadiusSearch.size(); ++i)
    {
      pt = cloudMap.points[pointIdxRadiusSearch[i]];
      localEUNMap.points.push_back(pt);
    }
  }
  localEUNMap.width = localEUNMap.points.size();
  localEUNMap.height = 1;
  localEUNMap.is_dense = true;

  pcl::toROSMsg(localEUNMap, localENUMap_pcd);
  localENUMap_pcd.header.frame_id = "world";
  local_ENU_map_pub.publish(localENUMap_pcd);

}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "map_generation");
  ros::NodeHandle n("~");

  drone_state_sub = n.subscribe("/drone_msg/drone_state", 50, drone_state_cb);

  local_ENU_map_pub = n.advertise<sensor_msgs::PointCloud2>("/local_obs_pcl", 1);

  n.param("map/resolution", _resolution, 0.2);
  n.param("sensing/sense_range", sense_range, 2.0);
  
  ros::Duration(1.0).sleep();

  MapGenerate();

  while (ros::ok())
  {
    ros::spinOnce();
  }
}


void MapGenerate()
{
  pcl::PointXYZ pt_random;  

    /* Obs 1 */
    {
    // Colomn xy position, r size, height
    double x, y, radius, h;
    x = 0;
    y = 0;
    radius = 0.05;
    h = 2;

    x = floor(x / _resolution) * _resolution + _resolution / 2.0;
    y = floor(y / _resolution) * _resolution + _resolution / 2.0;

    double last_x = x, last_y = y;
    for (int t = 0; t < ceil(h / _resolution); t++)
        for (double angle=0.0; angle<6.282; angle+=0.1)
        {
          double temp_x = x + (floor(radius*cos(angle)/_resolution)+0.5)*_resolution;
          double temp_y = y + (floor(radius*sin(angle)/_resolution)+0.5)*_resolution;
          if(abs(temp_x-last_x)<1e-3 && abs(temp_y-last_y)<1e-3){
          	continue;
          }else{
          	last_x = temp_x;
          	last_y = temp_y;
          }
          double temp_z = (t + 0.5) * _resolution;
          pt_random.x = temp_x;
          pt_random.y = temp_y;
          pt_random.z = temp_z;
          cloudMap.points.push_back(pt_random);
        }
    }
    
    /* Obs 2 */
    {
    // Colomn xy position, r size, height
    double x, y, radius, h;
    x = 0.0;
    y = 1.0;
    radius = 0.05;
    h = 2;

    x = floor(x / _resolution) * _resolution + _resolution / 2.0;
    y = floor(y / _resolution) * _resolution + _resolution / 2.0;

    double last_x = x, last_y = y;
    for (int t = 0; t < ceil(h / _resolution); t++)
        for (double angle=0.0; angle<6.282; angle+=0.1)
        {
          double temp_x = x + (floor(radius*cos(angle)/_resolution)+0.5)*_resolution;
          double temp_y = y + (floor(radius*sin(angle)/_resolution)+0.5)*_resolution;
          if(abs(temp_x-last_x)<1e-3 && abs(temp_y-last_y)<1e-3){
          	continue;
          }else{
          	last_x = temp_x;
          	last_y = temp_y;
          }
          double temp_z = (t + 0.5) * _resolution;
          pt_random.x = temp_x;
          pt_random.y = temp_y;
          pt_random.z = temp_z;
          cloudMap.points.push_back(pt_random);
        }
    }
    
    /* Obs 3 */
    {
    // Colomn xy position, r size, height
    double x, y, radius, h;
    x = 0.0;
    y = -1.0;
    radius = 0.05;
    h = 2;

    x = floor(x / _resolution) * _resolution + _resolution / 2.0;
    y = floor(y / _resolution) * _resolution + _resolution / 2.0;

    double last_x = x, last_y = y;
    for (int t = 0; t < ceil(h / _resolution); t++)
        for (double angle=0.0; angle<6.282; angle+=0.1)
        {
          double temp_x = x + (floor(radius*cos(angle)/_resolution)+0.5)*_resolution;
          double temp_y = y + (floor(radius*sin(angle)/_resolution)+0.5)*_resolution;
          if(abs(temp_x-last_x)<1e-3 && abs(temp_y-last_y)<1e-3){
          	continue;
          }else{
          	last_x = temp_x;
          	last_y = temp_y;
          }
          double temp_z = (t + 0.5) * _resolution;
          pt_random.x = temp_x;
          pt_random.y = temp_y;
          pt_random.z = temp_z;
          cloudMap.points.push_back(pt_random);
        }
    }
    
    /* Obs 4 */
    {
    // Colomn xy position, r size, height
    double x, y, radius, h;
    x = -1.0;
    y = -0.0;
    radius = 0.05;
    h = 2;

    x = floor(x / _resolution) * _resolution + _resolution / 2.0;
    y = floor(y / _resolution) * _resolution + _resolution / 2.0;

    double last_x = x, last_y = y;
    for (int t = 0; t < ceil(h / _resolution); t++)
        for (double angle=0.0; angle<6.282; angle+=0.1)
        {
          double temp_x = x + (floor(radius*cos(angle)/_resolution)+0.5)*_resolution;
          double temp_y = y + (floor(radius*sin(angle)/_resolution)+0.5)*_resolution;
          if(abs(temp_x-last_x)<1e-3 && abs(temp_y-last_y)<1e-3){
          	continue;
          }else{
          	last_x = temp_x;
          	last_y = temp_y;
          }
          double temp_z = (t + 0.5) * _resolution;
          pt_random.x = temp_x;
          pt_random.y = temp_y;
          pt_random.z = temp_z;
          cloudMap.points.push_back(pt_random);
        }
    }
    
    /* Obs 5 */
    {
    // Colomn xy position, r size, height
    double x, y, radius, h;
    x = 1.0;
    y = -0.0;
    radius = 0.05;
    h = 2;

    x = floor(x / _resolution) * _resolution + _resolution / 2.0;
    y = floor(y / _resolution) * _resolution + _resolution / 2.0;

    double last_x = x, last_y = y;
    for (int t = 0; t < ceil(h / _resolution); t++)
        for (double angle=0.0; angle<6.282; angle+=0.1)
        {
          double temp_x = x + (floor(radius*cos(angle)/_resolution)+0.5)*_resolution;
          double temp_y = y + (floor(radius*sin(angle)/_resolution)+0.5)*_resolution;
          if(abs(temp_x-last_x)<1e-3 && abs(temp_y-last_y)<1e-3){
          	continue;
          }else{
          	last_x = temp_x;
          	last_y = temp_y;
          }
          double temp_z = (t + 0.5) * _resolution;
          pt_random.x = temp_x;
          pt_random.y = temp_y;
          pt_random.z = temp_z;
          cloudMap.points.push_back(pt_random);
        }
    }
    
    /* Obs 6 */
    {
    // Colomn xy position, r size, height
    double x, y, radius, h;
    x = 1.0;
    y = -1.0;
    radius = 0.05;
    h = 2;

    x = floor(x / _resolution) * _resolution + _resolution / 2.0;
    y = floor(y / _resolution) * _resolution + _resolution / 2.0;

    double last_x = x, last_y = y;
    for (int t = 0; t < ceil(h / _resolution); t++)
        for (double angle=0.0; angle<6.282; angle+=0.1)
        {
          double temp_x = x + (floor(radius*cos(angle)/_resolution)+0.5)*_resolution;
          double temp_y = y + (floor(radius*sin(angle)/_resolution)+0.5)*_resolution;
          if(abs(temp_x-last_x)<1e-3 && abs(temp_y-last_y)<1e-3){
          	continue;
          }else{
          	last_x = temp_x;
          	last_y = temp_y;
          }
          double temp_z = (t + 0.5) * _resolution;
          pt_random.x = temp_x;
          pt_random.y = temp_y;
          pt_random.z = temp_z;
          cloudMap.points.push_back(pt_random);
        }
    }
    
    /* Obs 7 */
    {
    // Colomn xy position, r size, height
    double x, y, radius, h;
    x = -1.0;
    y = -1.0;
    radius = 0.05;
    h = 2;

    x = floor(x / _resolution) * _resolution + _resolution / 2.0;
    y = floor(y / _resolution) * _resolution + _resolution / 2.0;

    double last_x = x, last_y = y;
    for (int t = 0; t < ceil(h / _resolution); t++)
        for (double angle=0.0; angle<6.282; angle+=0.1)
        {
          double temp_x = x + (floor(radius*cos(angle)/_resolution)+0.5)*_resolution;
          double temp_y = y + (floor(radius*sin(angle)/_resolution)+0.5)*_resolution;
          if(abs(temp_x-last_x)<1e-3 && abs(temp_y-last_y)<1e-3){
          	continue;
          }else{
          	last_x = temp_x;
          	last_y = temp_y;
          }
          double temp_z = (t + 0.5) * _resolution;
          pt_random.x = temp_x;
          pt_random.y = temp_y;
          pt_random.z = temp_z;
          cloudMap.points.push_back(pt_random);
        }
    }
    
    /* Obs 8 */
    {
    // Colomn xy position, r size, height
    double x, y, radius, h;
    x = 1.0;
    y = 1.0;
    radius = 0.05;
    h = 2;

    x = floor(x / _resolution) * _resolution + _resolution / 2.0;
    y = floor(y / _resolution) * _resolution + _resolution / 2.0;

    double last_x = x, last_y = y;
    for (int t = 0; t < ceil(h / _resolution); t++)
        for (double angle=0.0; angle<6.282; angle+=0.1)
        {
          double temp_x = x + (floor(radius*cos(angle)/_resolution)+0.5)*_resolution;
          double temp_y = y + (floor(radius*sin(angle)/_resolution)+0.5)*_resolution;
          if(abs(temp_x-last_x)<1e-3 && abs(temp_y-last_y)<1e-3){
          	continue;
          }else{
          	last_x = temp_x;
          	last_y = temp_y;
          }
          double temp_z = (t + 0.5) * _resolution;
          pt_random.x = temp_x;
          pt_random.y = temp_y;
          pt_random.z = temp_z;
          cloudMap.points.push_back(pt_random);
        }
    }
    
    /* Obs 9 */
    {
    // Colomn xy position, r size, height
    double x, y, radius, h;
    x = -1.0;
    y = 1.0;
    radius = 0.05;
    h = 2;

    x = floor(x / _resolution) * _resolution + _resolution / 2.0;
    y = floor(y / _resolution) * _resolution + _resolution / 2.0;

    double last_x = x, last_y = y;
    for (int t = 0; t < ceil(h / _resolution); t++)
        for (double angle=0.0; angle<6.282; angle+=0.1)
        {
          double temp_x = x + (floor(radius*cos(angle)/_resolution)+0.5)*_resolution;
          double temp_y = y + (floor(radius*sin(angle)/_resolution)+0.5)*_resolution;
          if(abs(temp_x-last_x)<1e-3 && abs(temp_y-last_y)<1e-3){
          	continue;
          }else{
          	last_x = temp_x;
          	last_y = temp_y;
          }
          double temp_z = (t + 0.5) * _resolution;
          pt_random.x = temp_x;
          pt_random.y = temp_y;
          pt_random.z = temp_z;
          cloudMap.points.push_back(pt_random);
        }
    }
    
    /* Wall fence */
    {
    double wx1,wx2,wy3,wy4,h;
    wx1 = -2.2;
    wx2 = 2.2;
    wy3 = -2.2;
    wy4 = 2.2;
    h = 3;
    
    for(int t=0; t<ceil(h/_resolution);t++)
    {
      pt_random.z = (t + 0.5) * _resolution;
      for(int i=floor(wy3/_resolution); i<floor(wy4/_resolution);i++)
      {
        pt_random.x = wx1;
        pt_random.y = (i + 0.5) * _resolution;
        cloudMap.points.push_back(pt_random);
      }
      for(int i=floor(wy3/_resolution); i<floor(wy4/_resolution);i++)
      {
        pt_random.x = wx2;
        pt_random.y = (i + 0.5) * _resolution;
        cloudMap.points.push_back(pt_random);
      }
      for(int i=floor(wx1/_resolution); i<floor(wx2/_resolution);i++)
      {
        pt_random.x = (i + 0.5) * _resolution;
        pt_random.y = wy3;
        cloudMap.points.push_back(pt_random);
      }
      for(int i=floor(wx1/_resolution); i<floor(wx2/_resolution);i++)
      {
        pt_random.x = (i + 0.5) * _resolution;
        pt_random.y = wy4;
        cloudMap.points.push_back(pt_random);
      }
    }
    }

		
  cloudMap.width = cloudMap.points.size();
  cloudMap.height = 1;
  cloudMap.is_dense = true;

  ROS_WARN("Finished generate random map ");

  kdtreeLocalMap.setInputCloud(cloudMap.makeShared());

  _map_ok = true;
}

