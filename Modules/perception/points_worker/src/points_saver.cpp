#include<ros/ros.h>  
#include<pcl/point_cloud.h>  
#include<pcl_conversions/pcl_conversions.h>  
#include<sensor_msgs/PointCloud2.h>  
#include<pcl/io/pcd_io.h>  

using namespace std;

string pcd_file_save_at, pcl_topic_in;

void pcl_callback(const sensor_msgs::PointCloud2 &input)  
{  
  pcl::PointCloud<pcl::PointXYZ> cloud;  
  pcl::fromROSMsg(input, cloud);
  pcl::io::savePCDFileASCII (pcd_file_save_at, cloud);  
}  

main (int argc, char **argv)  
{  
  ros::init (argc, argv, "pcl_saver");  
  ros::NodeHandle nh("~"); 
  nh.param<string>("pcd_file_save_at", pcd_file_save_at, "~/Desktop/test.pcd");
  nh.param<string>("pcl_topic_in", pcl_topic_in, "/pcl_topic_in");
  ros::Subscriber bat_sub = nh.subscribe("/pcl_topic_in", 10, pcl_callback);  
  ros::spin();  
  return 0;  
} 
